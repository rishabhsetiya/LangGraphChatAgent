from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from mcp_tools import load_mcp_tools
from rag_tool import search_code
from langchain_core.messages import ToolMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_graph():
    # Groq via OpenAI-compatible API
    llm = ChatOpenAI(
        model=MODEL_ID,
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    # Tools
    mcp_tools = load_mcp_tools("http://localhost:8080/api/mcp")
    tools = mcp_tools + [search_code]

    llm_with_tools = llm.bind_tools(tools)

    # Nodes
    def llm_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    tool_node = ToolNode(tools)

    def route(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return "llm"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # Graph
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm",
        route,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph.add_edge("tools", "llm")

    return graph.compile()