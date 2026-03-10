from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph import graph
from langchain_core.tools import tool
import os,sys
import subprocess


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""

    print("yes LLM here")
    return f"The current weather in {location} is sunny with a high of 25°C."

# def apply_command(command: str):
#     try:
#         result = os.system(command = command)
#         return f"Command '{command}' executed with result: {result}"
#     except Exception as e:
#         return f"Error executing command '{command}': {str(e)}"
    
# apply_command("python3 openai_tool/openai_tool_graph.py")