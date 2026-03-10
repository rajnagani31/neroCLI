from google import genai
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
import time


class OpenAILLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"
        # self.client = genai.Client(api_key=self.api_key)
        print("--",self.api_key)

    def generate_response(self, prompt):
        start = time.time()
        model = ChatOpenAI(
            model=self.model,
        )

        messages = [("system", "You are a helpful assistant."),
                    ("user", prompt)]
        
        ai_msg = model.invoke(messages)
        end = time.time()
        print(ai_msg)
        print("Time taken:", end - start, "seconds")

OpenAILLMService().generate_response("hi, their")