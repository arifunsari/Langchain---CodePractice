from langchain_openai import ChatOpenAI  # Import OpenAI chat model
from langchain_core.prompts import PromptTemplate  # For prompt templating
from langchain_core.output_parsers import StrOutputParser  # To parse LLM output into string
from dotenv import load_dotenv  # Load environment variables like OpenAI API key
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough  # Runnable tools for chaining

load_dotenv()  # Load .env file for API keys

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',  # Prompt to generate a joke from topic
    input_variables=['topic']  # Expecting one variable: topic
)

model = ChatOpenAI()  # Load OpenAI chat model (e.g. GPT-3.5, GPT-4)

parser = StrOutputParser()  # Convert model output to plain string

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',  # Prompt to explain the joke
    input_variables=['text']  # Input will be the joke itself
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)  # Chain to generate joke: prompt → model → parse

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),  # Pass the joke as-is under the key 'joke'
    'explanation': RunnableSequence(prompt2, model, parser)  # Explain the joke using another chain
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)  # Full chain: generate joke, then explain it in parallel

print(final_chain.invoke({'topic':'cricket'}))  # Run the full chain with input topic 'cricket'


# jo input dege wahi wo output me return kar dega = RunnablePassthrought
# kaha use hota hai, jaise like LLM se ham ek topic return karwayege then us topic se joke
#  return karwaye ge then us joke ka explanation bhi, so hame explanation ke sath joke bhi chahiye then we use the pass through.