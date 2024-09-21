import streamlit as st
from dotenv import load_dotenv
from Rag.rag import generate_answer
from Rag.rag import get_conversational_chain
from Rag.rag import get_vectorstore
from Rag.rag import extract_text_from_pdf
from Rag.rag import text_chunk
from UI.html_templates import user_template, css, bot_template

# generating response from user queries and displaying them accordingly
def handle_question(question):
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(
        page_title = "Chat with Multiple pdf using RAG",
        page_icon = ":books:"
    )
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.headers("Chat with multiple pdf using RAG :book:")
    question = st.text_input("Ask the question from your documents: ")

    if question:
        handle_question(question= question)
    
    with st.sidebar:
        st.subheader("your documents")

        docs = st.file_uploader("upload your documents and click on 'process'", accept_multiple_file= True)

        if st.button("process"):
            with st.spinner("processing"):
                
                raw_text = extract_text_from_pdf(docs=docs)

                chunk = text_chunk(raw_text)
                
                vector = get_vectorstore(chunk)

                st.session_state  = get_conversational_chain(vector)




if __name__ == "__main__":
    main()
    