from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(
    model="models/gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Streamlit UI
st.header('🔬 Research Tool (Gemini)')

user_input = st.text_input('Enter your Prompt')

if st.button('Summarize'):
    if user_input:
        try:
            result = model.invoke(user_input)
            st.write(result.content)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a prompt before clicking Summarize.")
