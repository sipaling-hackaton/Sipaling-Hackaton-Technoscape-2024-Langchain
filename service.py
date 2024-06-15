import os
from pymongo import MongoClient
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai.chat_models import ChatOpenAI
import logging

logger = logging.getLogger("uvicorn")


async def get_references(client: MongoClient):
    db = client[os.getenv("APP_DB_NAME")]
    collection = db["Reference"]
    references = collection.find()
    return list(references)


def init_datasource(client: MongoClient, sources: list):
    db = client[os.getenv("DS_DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " "],
        length_function=len,
    )
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    data = []
    for source in sources:
        logger.info(f"Loading data from {source['url']}")
        is_source_exists = collection.find_one({"source": source["url"]})
        if is_source_exists is not None:
            logger.info(f"Source {source['url']} already exists in the collection")
            continue
        data.extend(WebBaseLoader(source["url"]).load())

    docs = text_splitter.split_documents(data)

    # reset the collection
    # collection.delete_many({})

    # insert the documents into the collection with the embeddings
    docsearch = MongoDBAtlasVectorSearch.from_documents(
        docs, embeddings, collection=collection, index_name=os.getenv("INDEX_NAME")
    )

    return docsearch


def gpt_ask(client: MongoClient, question: str):
    db = client[os.getenv("DS_DB_NAME")]
    col = db[os.getenv("COLLECTION_NAME")]
    vector_store = MongoDBAtlasVectorSearch(
        col,
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        index_name=os.getenv("INDEX_NAME"),
    )
    docs = vector_store.max_marginal_relevance_search(question, k=3)

    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are an intelligent assistant who helps answer customer questions. Input provided by customers is in the form of questions in various forms and can consist of several questions. You need to parse user input to select an existing question.

example question:
[4:51 PM, 6/13/2024] +62 895-0450-4930: Hello good afternoon, I want to ask. What is Teleperformance?
[4:52 PM, 6/13/2024] +62 895-0450-4930: What is the company's line of work?
then the customer's question is:
1. What is Teleperformance?
2. What is the field of work of the Teleperformance company?

After knowing the questions, answer each question based on the available evidence. If the evidence provided is still not supportive, just say you dont know. Return answers in a neat format and in the same language as entered.

customer input: {question} 
known facts: {context} 
possible answer:
    """,
    )

    combined_ctx = "\n\n".join([doc.page_content for doc in docs])

    model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=1, max_tokens=1000
    )
    llm_chain = LLMChain(llm=model, prompt=prompt_template)
    response = llm_chain.invoke(
        {
            "question": question,
            "context": combined_ctx,
        }
    )
    return response["text"]
