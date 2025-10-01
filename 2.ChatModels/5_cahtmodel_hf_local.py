from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Load a model using HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id='HuggingFaceH4/zephyr-7b-beta',  # fixed model name
    task='text-generation',
    pipeline_kwargs={
        'temperature': 0.5,
        'max_new_tokens': 100
    }
)

# Wrap it with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Invoke the model
result = model.invoke('What is the capital of India?')
print(result.content)
