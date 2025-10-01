from langchain_openai import ChatOpenAI  # OpenAI chat model
from langchain_core.prompts import PromptTemplate  # Prompt templating
from langchain_core.output_parsers import StrOutputParser  # Converts LLM output to string
from dotenv import load_dotenv  # Load environment variables from .env
from langchain.schema.runnable import RunnableSequence  # For chaining components

load_dotenv()  # Load API keys or other environment variables

# Prompt 1: Generate a joke based on a topic
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()  # Initialize the OpenAI model (e.g., GPT-3.5 or GPT-4)

parser = StrOutputParser()  # Parser to convert LLM output to plain string

# Prompt 2: Explain the joke using the generated joke as input
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Chain the full process:
# Step 1: Use prompt1 with input "topic"
# Step 2: Generate response with model
# Step 3: Parse joke to plain text
# Step 4: Feed joke to prompt2 (replacing {text})
# Step 5: Generate explanation using model again
# Step 6: Parse explanation to plain text
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# Run the chain with topic "AI"
print(chain.invoke({'topic':'AI'}))
