from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()  # Loads your .env file with the OpenAI API key

# Define the model
model = ChatOpenAI()

# Define output schema
class Review(TypedDict):
    summary: str
    sentiment: str

# Create structured model
structured_model = model.with_structured_output(Review)

# Input text to analyze
text = """
The hardware is great, but the software feels bloated. There are too
many pre-installed apps that I can't remove. Also, the UI looks outdated compared to
other brands. Hoping for a software update to fix this.
"""

# Invoke model with structured output
result = structured_model.invoke(text)

# Print results
print("Result:", result)
print("Summary:", result['summary'])
print("Sentiment:", result['sentiment'])
