#Open source model to genrate the embedding

from langchain_
hugginface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-tranformers/all-MiniLM-L6-v2')

text = 'Delhi is the capital of India'

vector =  embedding.embed_query(text)
print(str(vector))


# it will download the model of 90 mb approx
# output will be in  the embdding form of the text data = 384 Dimensional vector

# we can also genrate the embedding of document 
# # use for multiple line of text-  vector = embedding.embed_documents"
# use for single line of text- vector = embedding.embed_querry(text)