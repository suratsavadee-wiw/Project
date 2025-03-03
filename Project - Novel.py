# Databricks notebook source
# MAGIC %sh
# MAGIC cd /dbfs/FileStore/user/suratsavadee/
# MAGIC ls

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions

# COMMAND ----------

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat, AnalyzeDocumentRequest
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import (UnstructuredPDFLoader,
                                                  AzureAIDocumentIntelligenceLoader)
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough



import warnings
import nltk
import numpy as np
import pandas as pd
import re
import tiktoken
import os
import langchain
import io
import re

warnings.filterwarnings('ignore')

# COMMAND ----------

openai_api_version = "2024-02-01"
embedding_azure_deployment = "text-embedding-3-large"
llm_azure_deployment = "gpt-4o"
model_version = "2024-05-13"

llm = AzureChatOpenAI(
    openai_api_version=openai_api_version,
    azure_deployment=llm_azure_deployment,
    model_version=model_version,
    temperature=0
)

# COMMAND ----------

#file_path = "/dbfs/FileStore/user/suratsavadee/พรชีวัน.pdf"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Extract Text

# COMMAND ----------

file_path = "/dbfs/FileStore/user/suratsavadee/พรชีวัน.pdf"
endpoint = "your-endpoint"
key = "your-key"

layout_loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, 
    api_key=key, 
    file_path=file_path, 
    api_model="prebuilt-read", #prebuilt-layout
    mode= "page", #page #markdown
) 

# COMMAND ----------

docs = layout_loader.load()

# COMMAND ----------

type(docs)

# COMMAND ----------

len(docs)

# COMMAND ----------

print(f"Number of doc: {len(docs)}")

# COMMAND ----------

docs[:356]

# COMMAND ----------

docs[0]

# COMMAND ----------

type(docs[0]) #langchain

# COMMAND ----------

docs[0].page_content

# COMMAND ----------

novel_text = docs[0].page_content
print(novel_text)

# COMMAND ----------

print(f"Number of total characters: {len(novel_text):,}")

# COMMAND ----------

#txt saved in folder
save_path = "/dbfs/FileStore/user/suratsavadee/data/Novelfulltext.txt"

with open('save_path', 'r') as f:  
    full_text = f.read()

# COMMAND ----------

print(f"Number of total characters: {len(novel_text):,}")

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /dbfs/FileStore/user/suratsavadee/data/data
# MAGIC ls

# COMMAND ----------

# try regex pattern 
#pattern = r"..."

# Find all matches in the specified text
#matches = re.findall(pattern, text_to_search, flags=re.MULTILINE)

# Print the matched sentences
#for match in matches:
#    print(match)

# COMMAND ----------

# MAGIC %md
# MAGIC #Text Spliter

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# COMMAND ----------

docs_split = text_splitter.split_documents(docs)

# COMMAND ----------

type(docs_split)

# COMMAND ----------

print(text_splitter.split_documents(docs))

# COMMAND ----------

# MAGIC %md
# MAGIC #Embedding & Vector

# COMMAND ----------

#load document ==> split chunks
#Embedding ==> Embed chunks ==> vectors
#vector chunks ==> Save Chromadb
#"query" ==> similarlity search chromadb

# COMMAND ----------

embedding_azure_deployment = "text-embedding-3-large"
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=embedding_azure_deployment,
    openai_api_version=openai_api_version,
)

# COMMAND ----------

len(docs_split)

# COMMAND ----------

#try the minimum at 10 page first
chroma_db = Chroma.from_documents(docs_split[:10], embedding_model)

# COMMAND ----------

docs_similarity = print(docs_split[:10])

# COMMAND ----------

query = "นิยายเรื่องนี้มีตัวละครกี่คน"
search_docs = chroma_db.similarity_search(query)

# COMMAND ----------

print(search_docs[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC #Retriever

# COMMAND ----------

retrieval = chroma_db.as_retriever(search_type="similarity",
                                  k=2)

# COMMAND ----------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


system_prompt_template = """You are an assistant for question-answering tasks fot Thai Novel. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 

<Response Guidelines>
- Use three sentences maximum and keep the answer concise
- You MUST NOT mention something like "according to the document" or "context" in the answer
- You MUST answer in English if the question contains all English. You MUST answer in Thai if the question contains Thai
</Response Guidelines>

<Context>
{context}
</Context>"""

human_prompt_template = """<Question>
{question}
</Question>"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", human_prompt_template)
])

# COMMAND ----------

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_template
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retrieval, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# COMMAND ----------

chroma_db.as_retriever().invoke(query)[0]

# COMMAND ----------

langchain.debug = True

# COMMAND ----------

results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอกชื่ออะไร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "summarized the whole story for me"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "what happened at the middel of the story"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


system_prompt_template = """You are an assistant for question-answering tasks fot Thai Novel. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 

<Response Guidelines>
- Use three sentences maximum and keep the answer concise
- You MUST NOT mention something like "according to the document" or "context" in the answer
- You MUST answer in English if the question contains all English. You MUST answer in Thai if the question contains Thai
</Response Guidelines>

<Context>
{context}
</Context>"""

human_prompt_template = """<Question>
{question}
</Question>"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", human_prompt_template)
])
