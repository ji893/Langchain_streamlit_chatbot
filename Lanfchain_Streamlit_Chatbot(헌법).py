import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader  # 임포트 수정
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#PDF 파일 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

#Document 객체를 벡터 DB에 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./FAISS_db"
    vectorstore = FAISS.from_documents(
        split_docs, 
        OpenAIEmbeddings(model='text-embedding-3-small')
    )
    vectorstore.save_local(persist_directory)
    return vectorstore


#만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./FAISS_db"
    if os.path.exists(persist_directory):
        return FAISS.load_local(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


#document 객체의 page_content 를 join


@st.cache_resource
def chaining():
    file_path =r"C:\RAG_Project\대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    Please answer in Korean and use respectful language.\
    {context}
    """

    qa_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model = "gpt-4o")
    rag_chain = (
        {"context": lambda x: format_docs(retriever.invoke(x)), "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

st.header("헌법 Q&A 챗봇🧑‍⚖️")
rag_chain = chaining()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

prompt_message = st.chat_input("질문을 입력해주세요 :)")

# 메시지 업데이트
if prompt_message:
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(prompt_message)
    st.session_state.messages.append({"role": "assistant", "content": response})

# 메시지 출력 (항상 마지막에만 렌더링)
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])