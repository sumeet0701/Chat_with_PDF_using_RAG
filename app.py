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
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    
    st.header("Chat with multiple PDFs :books:")
    question=st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                #get the pdf
                raw_text=extract_text_from_pdf(docs)
                
                #get the text chunks
                text_chunks=text_chunk(raw_text)
                
                #create vectorstore
                vectorstore=get_vectorstore(text_chunks)
                
                #create conversation chain
                st.session_state.conversation=get_conversational_chain(vectorstore)


if __name__ == '__main__':
    main()