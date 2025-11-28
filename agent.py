from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # <<< REQUIRED FIX

RECURSION_LIMIT = 5000

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]

# -------------------------------------------------
# GEMINI LLM INITIALIZATION (FIXED)
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,
    check_every_n_seconds=1,
    max_bucket_size=9
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,              # <<< FIXED: Added API key
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint.
3. Solve the task exactly as required.
4. Submit the answer ONLY to the endpoint specified on the current page (never make up URLs).
5. Read the server response and:
   - If it contains a new quiz URL → fetch it immediately and continue.
   - If no new URL is present → return "END".

STRICT RULES:

GENERAL RULES:
- NEVER stop early.
- NEVER hallucinate URLs.
- NEVER shorten URLs.
- ALWAYS follow the server-provided flow.
- ALWAYS continue until no new URL is provided.

TIME LIMIT:
- Each task has a 3-minute limit.
- If answer is wrong → retry.

STOPPING CONDITION:
- Only return END when the server response contains no new URL.

ADDITIONAL INFORMATION:
- Email: {EMAIL}
- Secret: {SECRET}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm

# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}

# -------------------------------------------------
# ROUTING LOGIC
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]

    # Support for tool calls
    tool_calls = getattr(last, "tool_calls", None) if hasattr(last, "tool_calls") else last.get("tool_calls", None)
    if tool_calls:
        return "tools"

    # Read content
    content = getattr(last, "content", None) if hasattr(last, "content") else last.get("content")

    # End condition
    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and content[0].get("text", "").strip() == "END":
        return END

    return "agent"

# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

app = graph.compile()

# -------------------------------------------------
# ENTRY FUNCTION
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke(
        {"messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed succesfully")
