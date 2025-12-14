import streamlit as st
from app import  create_vector_database, question
import uuid


st.set_page_config(page_title="Peril_In_Pinebrook", layout="centered")

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

thread_id = st.session_state["thread_id"]
config = {"configurable": {"thread_id": thread_id}}

st.title(":red[Peril_In_Pinebrook]")
st.caption(f"Thread ID: {thread_id}")
# Button to create the vector database
if st.button("create_database"):
    create_vector_database()
    
if st.button("Clear Chat"):
    st.session_state["chat_history"] = []  # Clears the chat history
    
# Initialize the session state to store chat history if it doesn't already exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Container for chat messages
messages = st.container(height=500)
with messages:
    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            messages.chat_message("user", avatar="🧔🏽‍♂️").write(f"**User:** {chat['message']}")
        else:
            messages.chat_message("assistant", avatar="🧙🏻‍♂️").write(f"**AI:** {chat['message']}")

# Chat input for user prompts
if prompt := st.chat_input("Say something"):
    # Append user's message to chat history and display it immediately
    st.session_state["chat_history"].append({"role": "user", "message": prompt})
    messages.chat_message("user", avatar="🧔🏽‍♂️").write(f"**User:** {prompt}")
    
    # Generate AI response
    chain = question(config=config)
    app = chain.invoke({"input": prompt},config=config)
    response = app["answer"]
    
    # Append AI response to chat history and display it
    st.session_state["chat_history"].append({"role": "ai", "message": response})
    messages.chat_message("assistant", avatar="🧙🏻‍♂️").write(f"**AI:** {response}")
