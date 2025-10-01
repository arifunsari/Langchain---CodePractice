from langchain_openai import ChatOpenAI
from dotenv import_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

result = model.invoke("What is the captial of India")
print(result.content)