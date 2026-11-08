import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.title("Friendly Book Finder")
st.write("Describe the kind of book you want, and I'll find matches from my data")


#initiatlize embedding and vectore store
@st.cache_resource
def load_document():
    loader = CSVLoader(
        file_path= "books.csv",
        encoding= "utf-8",
        source_column= "url",
        content_columns= ["text"],
        metadata_columns= ["title","category","price","rating","availability"]
    )
    return loader.load()

@st.cache_resource
def load_vector_stores():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    persist_direct = "books_db"
    store = Chroma(
        persist_directory=persist_direct,
        embedding_function=embeddings,
        collection_name= "BOOKS"
    )

    if store._collection.count()==0:
        docs=load_document()
        store.add_documents(docs)

    return store.as_retriever(search_kwargs = {"k":5})

retrievers = load_vector_stores()

#initialize llm model
llm = ChatGroq(model = "qwen/qwen3.8-27b", temperature=0.2)

#Prompts and chains
system = """Your work is to select the books as per the users requirements with the price,rating, availability and summarize description. 
if the question asked is not related than politely reply as not related query please try again. and suggest random 5 books.
Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human","{input}"),
])

document_stuff_chain = create_stuff_documents_chain(llm= llm, prompt=prompt)
chain = create_retrieval_chain(retrievers,document_stuff_chain)

# Streamlit app
if "chat" not in st.session_state:
    st.session_state.chat=[]

for who,text in st.session_state.chat:
    st.chat_message(who).write(text)

q =  st.chat_input("Ask about any book or recomendation")

if q:
    st.chat_message("user").write(q)
    st.session_state.chat.append(("user",q))

    with st.spinner("Searching book ......."):
        response = chain.invoke({"input":q})
        bot_reply = response["answer"]

    st.chat_message("assistant").write(bot_reply)
    st.session_state.chat.append(("assistant",bot_reply))