from langchain_openai import ChatOpenAI  # Import OpenAI chat model
from langchain_core.prompts import PromptTemplate  # Import prompt templating class
from langchain_core.output_parsers import StrOutputParser  # Converts model output to plain string
from dotenv import load_dotenv  # Loads environment variables from .env file
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel  # Core components for chaining

load_dotenv()  # Load environment variables (like API keys)

# Define a custom function to count the number of words in a string
def word_count(text):
    return len(text.split())

# Create a prompt to generate a joke based on a given topic
prompt = PromptTemplate(
    template='Write a joke about {topic}',  # Template with placeholder for topic
    input_variables=['topic']  # 'topic' is the input variable
)

model = ChatOpenAI()  # Initialize the chat model (e.g. GPT-4)

parser = StrOutputParser()  # Parser to convert the LLM output to a simple string

# Chain to generate a joke: prompt → model → output parser
joke_gen_chain = RunnableSequence(prompt, model, parser)

# Parallel chain to pass the joke and calculate its word count simultaneously
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),  # Returns the joke as-is
    'word_count': RunnableLambda(word_count)  # Applies the word_count function to the joke
})

# Final chain: generate a joke first, then compute joke and word count in parallel
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run the final chain with input topic 'AI'
result = final_chain.invoke({'topic': 'AI'})

# Format the final output to show the joke and its word count
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# Print the result
print(final_result)

#  Summary:
# The code generates a joke about "AI", then:

# Returns the joke.

# Computes the word count of the joke using a custom function via RunnableLambda.

# Finally, it prints both the joke and the word count in a formatted string.
