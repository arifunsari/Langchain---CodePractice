from langchain_openai import OpenAI
from dotenv import_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the captial of India")

print(result)