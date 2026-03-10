# openai_adapter/openai_llm_service.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
import time

load_dotenv()

class OpenAILLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"
        if self.api_key is None:
            raise ValueError("OPENAI_API_KEY is not found")
        # Store bound ChatOpenAI instance
        self._chat_model = None

    def bind_tools(self, tools):
        """Bind tools to the model (CRITICAL for tool calling)"""
        self._chat_model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key
        ).bind_tools(tools)
        return self  # Return self for chaining

    def invoke(self, messages: list[BaseMessage]):
        """LangGraph expects .invoke() method"""
        if not self._chat_model:
            raise ValueError("Must call bind_tools() first")
        return self._chat_model.invoke(messages)

    def generate_response(self, query, instructions, tools):
        """Your original method (kept for backward compatibility)"""
        print("Generating response from OpenAI...")
        print(f"Query: {query}")
        print(f"Instructions: {instructions}")
        print(f"Tools: {tools}")

        start = time.time()
        model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key
        ).bind_tools(tools) if tools else ChatOpenAI(
            model=self.model, api_key=self.api_key
        )

        messages = [("system", instructions), ("user", query)]
        ai_msg = model.invoke(messages)
        end = time.time()
        print(ai_msg)
        print("Time taken:", end - start, "seconds")
        return "ai_msg"