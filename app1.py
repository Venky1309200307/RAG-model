import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

mongo = MongoClient(
    "mongodb+srv://venkypranee13:sXCfkJBiP_66t5@cluster0.xnwx1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&tlsAllowInvalidCertificates=true",
)

def process_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    print(len(chunks))

    doc_embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        google_api_key="AIzaSyBT_cXS1-V5ggaDcx7heSHJMb0h1r-xoPU",
    )
    dbName = "Final_year"
    collectionName = "Research_paper_embeddings"
    collection = mongo[dbName][collectionName]

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=doc_embeddings_model,
        collection=collection,
        index_name="embedding",
    )    
    return True

def chat(Question):

    question = Question
    doc_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key="AIzaSyBT_cXS1-V5ggaDcx7heSHJMb0h1r-xoPU",
    )

    dbName = "Final_year"
    collectionName = "Research_paper_embeddings"

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    "mongodb+srv://venkypranee13:sXCfkJBiP_66t5@cluster0.xnwx1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&tlsAllowInvalidCertificates=true",
    dbName + "." + collectionName,
    doc_embeddings_model,
    index_name="embedding",
    )
    prompt_template = """Interact with the user based upon their sentiment and check whether the context is related to a research paper, if it is a research paper, then answer to the questions asked by the user. If not ask the user to upload a research paper and you neednt answer the question asked by the user.

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )

    chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyB3BBf69PnHSy1crohfyymSJfDmvLdRjvs",
    )

    output_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_search.as_retriever(search_kwargs={"k": 110})

    RAG_Chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | chat_model
        | output_parser
    )

    response = RAG_Chain.invoke(question)
    return response

def main():
    st.set_page_config(page_title="Paper Pulse", page_icon=":books:")
    
    if "is_uploaded" not in st.session_state:
        st.session_state.is_uploaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Your documents")
        UPLOAD_FOLDER = 'uploads'
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        uploaded_file = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                if uploaded_file is not None:
                    file_name = uploaded_file.name
                    file_path = os.path.join(UPLOAD_FOLDER, file_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.session_state.is_uploaded = process_file(file_path)

    if st.session_state.is_uploaded:

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question here...."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner(text="Hold on, I'm generating content for you😉..."):
                    stream = chat(prompt)
                    st.write(stream)
            st.session_state.messages.append({"role": "assistant", "content": stream})

if __name__ == '__main__':
    main()