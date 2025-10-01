from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books', # directory ka path
    glob='*.pdf', # is book folder ke ander saare pdf file ko load kar rahe ho.
    loader_cls=PyPDFLoader  # loader ka class batana hota hai jo pypdfloader hai.
)

# docs = loader.load() # when we want to load everything , at once it take time to load it is slow., use when no. of docs is small.
# print(len(docs))


# print(docs[325].page_content)  # load the last page content and meta data
# print(docs[325].metadata)

docs = loader.lazy_load()  # jab large file ko load krna ho then we use the lazy_load(), load on the demand.

for document in docs:
    print(document.metadata)