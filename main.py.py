import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")
    st.image('yumchang.png')
    st.title("_염창중학교 :red[Q&A]_ 🏫")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        folder_path = Path()  # 'Path' 객체를 생성하여 폴더 경로를 설정
        files_text = get_text_from_folder(folder_path)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        model_name = st.radio(
        "Select your model",
        ['gpt-4', 'gpt-3.5-turbo']
    )
        process = st.button("Process")
    

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text_from_folder(folder_path)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key, model_name) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_from_folder(folder_path):

    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file_path in files:
        if file_path.is_file():
            file_name = file_path.name
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_name.endswith('.docx'):
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_name.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(str(file_path))
                documents = loader.load_and_split()
            else:
                documents = []  # PDF, DOCX, PPTX 파일이 아닌 경우, 'documents'를 빈 리스트로 설정
            doc_list.extend(documents)
    return doc_list


def get_text_from_file(file_path):
    doc_list = []

    file_name = file_path
    loader = PyPDFLoader(file_name)
    documents = loader.load_and_split()
    doc_list.extend(documents)

    return doc_list

def get_text(folder_path):
    doc_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load_and_split()
            elif file_name.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents = loader.load_and_split()
            elif file_name.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file_path)
                documents = loader.load_and_split()
            doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = model_name ,temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
