# openai_tool/openai_tool_graph.py

import os
import sys

_here = os.path.dirname(__file__)
_pkg_root = os.path.abspath(os.path.join(_here, '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from openai_tools import get_current_weather  # Your @tool
from openai_adapter.openai_llm_service import OpenAILLMService

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 👈 CRITICAL: Bind tools to LLM FIRST
tools = [get_current_weather]
llm = OpenAILLMService().bind_tools(tools)  # This sets up tool callingAIMessage

def agent_response(state: State):
    """Agent: LLM decides which tools to call"""
    print("🧠 1. Agent is generating response...")
    messages = state["messages"]
    print("Current conversation messages:", messages)

    if messages[-1].content.lower() in ['hi', 'hello','HI','Hello','hi, thier']:
        return {"messages": [AIMessage(content="Hello! How can I assist you today?!!")]}
    response = llm.invoke(messages)  # Now works!
    print("Agent response:", response)
    return {"messages": [response]}

def execute_tools(state: State):
    """Execute the tool calls from agent"""
    print("🔧 3. Executing tools called by agent...")
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        # Execute tool
        for t in tools:
            if t.name == tool_name:
                result = t.invoke(tool_args)
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call['id'],
                    name=tool_name
                ))
                print(f"✅ Tool executed: {tool_name}({tool_args}) -> {result}")
                break
    
    return {"messages": tool_messages}

def should_continue(state: State) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    print("🔍 2. Checking if agent called tools...")
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(State)

# Node
graph.add_node("agent_response", agent_response)
graph.add_node("tools", execute_tools)

# starts and edges
graph.set_entry_point("agent_response")
graph.add_conditional_edges(
    "agent_response", should_continue,
    {
        "tools": "tools",
        END: END
        }
)
graph.add_edge("tools", "agent_response")  # Loop back to agent after tools

# Compile the graph into an app
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="HI")],
    })
    print("Final response:", result["messages"][-1].content)