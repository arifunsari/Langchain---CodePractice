from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create embedding object correctly
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32  # <-- Pass directly, not inside model_kwargs
)

# Embed the query
result = embedding.embed_query("Delhi is the capital of India")

# Print the result
print(result)
