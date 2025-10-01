import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# --- Streamlit UI Setup ---
st.set_page_config(page_title="YouTube Podcast Analyzer", layout="wide")
st.title("🎙️ YouTube Podcast Analyzer with Gemini")
st.markdown("""
    This app allows you to:
    1. Add your own **Gemini API Key**
    2. Enter a **YouTube video link** to extract the transcript
    3. Ask **questions** about the transcript
""")

# --- Gemini API Key Section ---
st.sidebar.header("🔐 Gemini API Key")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
else:
    st.warning("Please enter your Gemini API key in the sidebar to continue.")
    st.stop()

# --- YouTube Video Input Section ---
st.header("📺 Video Transcript Fetcher")
youtube_url = st.text_input("Enter YouTube Video URL (e.g. https://youtu.be/v-jPiFqTOsg)")

video_id = ""
if youtube_url:
    try:
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            st.error("❌ Invalid YouTube URL")
    except:
        st.error("❌ Unable to extract video ID")

    transcript = ""
    if video_id:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            st.success("✅ Transcript fetched successfully!")
            with st.expander("📜 View Transcript"):
                st.write(transcript)
        except TranscriptsDisabled:
            st.error("❌ Transcript not available for this video.")
            st.stop()

        # --- Chunk the transcript ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # --- Embedding & Vector Store ---
        with st.spinner("🔎 Generating embeddings and storing in vector DB..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # --- Prompt Template ---
        prompt_template = PromptTemplate(
            template="""
You are a helpful assistant. Answer the question only using the transcript context.
If the context is not enough, just say \"I don't know\".

{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        # --- Q&A Section ---
        st.header("💬 Ask Questions About the Video")
        question = st.text_area("Enter your question")

        if st.button("Get Answer") and question:
            llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

            with st.spinner("💡 Thinking..."):
                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                final_prompt = prompt_template.format(context=context_text, question=question)
                response = llm.invoke(final_prompt)
                st.success("📢 Answer:")
                st.write(response.content)

        st.markdown("---")
        st.info("Tip: Try questions like 'What was discussed about AI?', 'Summarize the main ideas' etc.")

else:
    st.info("Paste a YouTube link to begin.")



##__________________________________--------------------------------______________________________________---------------------------------
# import streamlit as st
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv

# # 📄 Set Streamlit app page configuration
# st.set_page_config(page_title="YouTube Podcast Analyzer", layout="wide")

# # 🧠 App Title and Intro Description
# st.title("🎙️ YouTube Podcast Analyzer with Gemini")
# st.markdown("""
#     This app allows you to:
#     1. Add your own **Gemini API Key**
#     2. Enter a **YouTube video link** to extract the transcript
#     3. Ask **questions** about the transcript
# """)

# # 🔐 Sidebar: Get Gemini API key from user
# st.sidebar.header("🔐 Gemini API Key")
# gemini_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

# # ✅ Set the environment variable for Gemini API
# if gemini_api_key:
#     os.environ["GOOGLE_API_KEY"] = gemini_api_key
# else:
#     st.warning("Please enter your Gemini API key in the sidebar to continue.")
#     st.stop()  # Stop the app if key is not provided

# # 📺 Section to input YouTube URL
# st.header("📺 Video Transcript Fetcher")
# youtube_url = st.text_input("Enter YouTube Video URL (e.g. https://youtu.be/v-jPiFqTOsg)")

# video_id = ""
# # 🔍 Extract YouTube video ID from URL
# if youtube_url:
#     try:
#         if "v=" in youtube_url:
#             video_id = youtube_url.split("v=")[1].split("&")[0]
#         elif "youtu.be/" in youtube_url:
#             video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
#         else:
#             st.error("❌ Invalid YouTube URL")
#     except:
#         st.error("❌ Unable to extract video ID")

#     transcript = ""
#     # 📝 Fetch transcript if video ID is available
#     if video_id:
#         try:
#             transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
#             transcript = " ".join(chunk["text"] for chunk in transcript_list)
#             st.success("✅ Transcript fetched successfully!")
#             with st.expander("📜 View Transcript"):
#                 st.write(transcript)  # 📜 Show full transcript in collapsible section
#         except TranscriptsDisabled:
#             st.error("❌ Transcript not available for this video.")
#             st.stop()

#         # ✂️ Split transcript into manageable chunks
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.create_documents([transcript])

#         # 🔐 Generate embeddings and store in FAISS vector DB
#         with st.spinner("🔎 Generating embeddings and storing in vector DB..."):
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             vector_store = FAISS.from_documents(chunks, embeddings)
#             retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#         # 📜 Prompt template to control LLM behavior
#         prompt_template = PromptTemplate(
#             template="""
# You are a helpful assistant. Answer the question only using the transcript context.
# If the context is not enough, just say \"I don't know\".

# {context}

# Question: {question}
# """,
#             input_variables=["context", "question"]
#         )

#         # ❓ Ask Question Section
#         st.header("💬 Ask Questions About the Video")
#         question = st.text_area("Enter your question")  # 🧾 Input box for question

#         # ▶️ On clicking "Get Answer", fetch response
#         if st.button("Get Answer") and question:
#             llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)  # 🤖 Load Gemini LLM

#             with st.spinner("💡 Thinking..."):
#                 retrieved_docs = retriever.invoke(question)  # 🔍 Get top relevant transcript chunks
#                 context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)  # 🧩 Combine chunks into context
#                 final_prompt = prompt_template.format(context=context_text, question=question)  # 🧠 Fill prompt with data
#                 response = llm.invoke(final_prompt)  # 🤖 Generate answer
#                 st.success("📢 Answer:")
#                 st.write(response.content)  # 📄 Display answer

#         st.markdown("---")
#         st.info("Tip: Try questions like 'What was discussed about AI?', 'Summarize the main ideas' etc.")

# else:
#     st.info("Paste a YouTube link to begin.")  # ℹ️ Ask user to paste URL to proceed
