import streamlit as st
from app import get_qa_chain, create_vector_database

st.set_page_config(page_title="Werewolf Q&A", layout="centered")


st.title(":green[Werewolf Q&A] 🐺")

# Button to create the vector database
if st.button("🔒 create_database"):
    create_vector_database()
    
if st.button("🗑️ Clear Chat"):
    st.session_state["chat_history"] = []  # Clears the chat history

# Initialize the session state to store chat history if it doesn't already exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Container for chat messages
messages = st.container(height=500)
with messages:
    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            messages.chat_message("user").write(f"**User:** {chat['message']}")
        else:
            messages.chat_message("assistant").write(f"**AI:** {chat['message']}")

# Chat input for user prompts
if prompt := st.chat_input("Say something"):
    # Append user's message to chat history and display it immediately
    st.session_state["chat_history"].append({"role": "user", "message": prompt})
    messages.chat_message("user").write(f"**User:** {prompt}")
    
    # Generate AI response
    chain = get_qa_chain()
    response = chain.invoke(prompt)
    
    # Append AI response to chat history and display it
    st.session_state["chat_history"].append({"role": "ai", "message": response})
    messages.chat_message("assistant").write(f"**AI:** {response}")
