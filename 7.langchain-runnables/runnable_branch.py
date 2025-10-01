from langchain_openai import ChatOpenAI # LLM model 
from langchain_core.prompts import PromptTemplate  # for  giving the prompt 
from langchain_core.output_parsers import StrOutputParser # return the output in the strin formata only
from dotenv import load_dotenv  # import the load_dotenv to keep .env file to secure the API
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda # runnable function to exectuer code 
# smoothly just coonector

load_dotenv() #load the .env api key or files 

prompt1 = PromptTemplate( # prompt1  
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate( #prompt2 
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI() # initialize the LLM default one liek gpt-3

parser = StrOutputParser() # return the output in plain string 

report_gen_chain = prompt1 | model | parser # now create a chain 

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))



