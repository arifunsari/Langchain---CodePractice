# 1️⃣ INSTALL REQUIRED LIBRARIES (run this in a cell or terminal if not already done)
# pip install youtube-transcript-api langchain-community langchain-google-genai \
#             faiss-cpu tiktoken python-dotenv google-generativeai==0.8.5

# 2️⃣ IMPORT MODULES
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# 3️⃣ LOAD .env AND API KEY
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 4️⃣ FETCH YOUTUBE TRANSCRIPT
video_id = "v-jPiFqTOsg"  # Replace with your video ID

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("✅ Transcript fetched.")
except TranscriptsDisabled:
    print("❌ No captions available for this video.")
    transcript = ""

# 5️⃣ SPLIT TEXT INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript]) if transcript else []

# 6️⃣ EMBEDDING + VECTOR DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

# 7️⃣ RETRIEVER SETUP
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 8️⃣ PROMPT TEMPLATE
prompt = PromptTemplate(
    template="""
You are a helpful assistant. Answer the question only using the transcript context.
If the context is not enough, just say "I don't know".

{context}

Question: {question}
""",
    input_variables=["context", "question"],
)

# 9️⃣ LLM SETUP
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

# 🔟 ASK A QUESTION
question = "Summarize the topic talked realted about the Data Science"

# Vector similarity search
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Format the final prompt
final_prompt = prompt.format(context=context_text, question=question)

# Get answer
response = llm.invoke(final_prompt)

print("📢 Answer:")
print(response.content)
