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
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
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
import time

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

# file_path = "/dbfs/FileStore/user/suratsavadee/พรชีวัน.pdf"
# endpoint = "your-endpoint"
# key = "your-key"

# layout_loader = AzureAIDocumentIntelligenceLoader(
#     api_endpoint=endpoint, 
#     api_key=key, 
#     file_path=file_path, 
#     api_model="prebuilt-read", #prebuilt-layout model
#     api_version="2024-02-29-preview",
#     mode= "single", #page #markdown 
# )

# COMMAND ----------

#e
file_path = "/dbfs/FileStore/user/suratsavadee/พรชีวัน.pdf"
# endpoint = "your-endpoint"
# key = "your-key"

layout_loader = AzureAIDocumentIntelligenceLoader(
api_endpoint=endpoint, 
    api_key=key, 
    file_path=file_path, 
    api_model="prebuilt-read", #prebuilt-layout model
    api_version="2024-02-29-preview",
    mode= "single", #page #markdown 
)

# COMMAND ----------

docs = layout_loader.load()

# COMMAND ----------

type(docs)

# COMMAND ----------

len(docs)

# COMMAND ----------

type(docs[0]) #metadata or #pagecontent

# COMMAND ----------

docs[0].page_content

# COMMAND ----------

# docs[0].metadata

# COMMAND ----------

all_text = docs[0].page_content
save_path = "/dbfs/FileStore/user/suratsavadee/novel.txt"
print(f"Number of characters: {len(all_text):,}")
with open(save_path, "w") as f:
  f.write(all_text)

# COMMAND ----------

# MAGIC %md
# MAGIC #Text Spliter

# COMMAND ----------

checkpoint_path = "/dbfs/FileStore/user/suratsavadee/novel.txt"
with open(checkpoint_path, "r") as f:
  all_text = f.read()
print(f"Number of characters: {len(all_text):,}")

# COMMAND ----------

type(all_text)

# COMMAND ----------

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len,
# )

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
)

# COMMAND ----------

text_lst = text_splitter.split_text(all_text)

# COMMAND ----------

type(text_lst)

# COMMAND ----------

len(text_lst)

# COMMAND ----------

# gpt-4 and text-embedding-3-large have same encoding
# check number of characters
encoding = tiktoken.encoding_for_model("gpt-4")

stats = [{"num_char": len(t),
          "num_token": len(encoding.encode(t))} for t in text_lst]

stats_df = pd.DataFrame(stats)
print(stats_df.head())

# COMMAND ----------

stats_df.describe()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.histplot(data=stats_df, x="num_token", binwidth=500)
ax.set_xticks(range(0, 5000, 500))
ax.grid(axis="both", linewidth=0.5, alpha=0.5)
ax.set_title("Distribution of document lengths in the knowledge base (in count of tokens)")

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

# persist_directory = 'docs/'

# vectordb = Chroma(embedding_function=embedding_model,
#                  persist_directory=persist_directory)

# COMMAND ----------

persist_directory = 'docs_15k_chunk/'

vectordb2 = Chroma(embedding_function=embedding_model,
                 persist_directory=persist_directory)

# COMMAND ----------

vectordb2._collection.count()

# COMMAND ----------

vectordb2.add_texts(text_lst[:10])

# COMMAND ----------

vectordb2._collection.count()

# COMMAND ----------

# vectordb.add_texts(text_lst[:10])

# COMMAND ----------

from tqdm.notebook import tqdm


for i in tqdm(range(10, len(text_lst), 50)):
  start = i
  end = i + 50
  print(f"{start} - {end}")
  subset = text_lst[start:end]
  vectordb2.add_texts(subset)

# COMMAND ----------

vectordb2._collection.count()

# COMMAND ----------

len(text_lst)

# COMMAND ----------

# load from disk
persist_directory = 'docs_15k_chunk/'
temp_db = Chroma(collection_name="langchain",
                 persist_directory=persist_directory, 
             embedding_function=embedding_model)

# COMMAND ----------

temp_db._collection.count()

# COMMAND ----------

docs_similarity = print(text_lst)

# COMMAND ----------

#retrieval = temp_db.as_retriever(search_type="similarity",
                                  #k=5)

# COMMAND ----------

retrieval = temp_db.as_retriever( search_type="mmr",
                                  search_kwargs={'k': 7, 'fetch_k': 50})

# COMMAND ----------

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# system_prompt_template = """You are an assistant for question-answering tasks for Thai Novel. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. If you unsure the answer, just say you unsure. 

# <Response Guidelines>
# - Use three sentences maximum and keep the answer concise
# - You MUST NOT mention something like "according to the document" or "context" in the answer
# - You MUST answer in English if the question contains all English. You MUST answer in Thai if the question contains Thai
# </Response Guidelines>

# <Context>
# {context}
# </Context>"""

# human_prompt_template = """<Question>
# {question}
# </Question>"""

# prompt_template = ChatPromptTemplate.from_messages([
#    ("system", system_prompt_template),
#   ("human", human_prompt_template)
# ])

# COMMAND ----------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


system_prompt_template = """You are an assistant for question-answering tasks for Thai Novel. Use the following pieces of retrieved context to answer the question. If you don't know or unsure the answer, just say that you don't know or unsure.

<Response Guidelines>
- You MUST provide the exact answer followed by scenary to support the answer 
- You MUST NOT mention something like "according to the document" or "context" in the answer
- You MUST answer in Thai language
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

#results = rag_chain_with_source.invoke(query)

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

temp_db.as_retriever().invoke(query)[0]

# COMMAND ----------

langchain.debug = True

# COMMAND ----------

search_docs = temp_db.similarity_search(query)

# COMMAND ----------

#query = "ชีวันเป็นลูกใคร"
#search_docs = temp_db.similarity_search(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "ชีวันเป็นลูกใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "พรชีวาเป็นลูกใคร"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวาเป็นลูกใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "พระเอก นางเอกคือใคร"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอก นางเอกคือใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอก นางเอกคือใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอก นางเอกคือใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอก นางเอกคือใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พระเอก นางเอกคือใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "พรชีวาคู่กับใคร"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวาคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "ความสัมพันธ์ระหว่างพรชีวากับรณภูมิ"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "ความสัมพันธ์ระหว่างพรชีวากับรณภูมิ"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "พรชีวากับรณภูมิคู่กันไหม"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวากับรณภูมิคู่กันไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวากับรณภูมิคู่กันไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวากับรณภูมิคู่กันไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

#query = "สรุจกับพรชีวารักกันตอนไหน"
#results = rag_chain_with_source.invoke(query)

# COMMAND ----------

#print(f"Question: {results['question']}\n"
      #f"Answer: {results['answer']}")

# COMMAND ----------

query = "สรุจกับพรชีวันรักกันตอนไหน"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "สรุจกับพรชีวันรักกันตอนไหน"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "สรุจกับพรชีวันรักกันตอนไหน"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

# query = "สรุปเรื่องนี้ให้หน่อย"
# results = rag_chain_with_source.invoke(query)

# COMMAND ----------

# print(f"Question: {results['question']}\n"
#       f"Answer: {results['answer']}")

# COMMAND ----------

# query = "สรุปเรื่องนี้ให้หน่อย"
# results = rag_chain_with_source.invoke(query)

# COMMAND ----------

# print(f"Question: {results['question']}\n"
#       f"Answer: {results['answer']}")

# COMMAND ----------

# query = "ย่อเรื่องนี้ให้หน่อย"
# results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "สรุจคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "สรุจคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวาคู่กับรณภูมิใช่ไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวาคู่กับรณภูมิใช่ไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "พรชีวาคู่กับรณภูมิใช่ไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "รณภูมิคู่ใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "รณภูมิคู่ใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

print(f"Question: {results['question']}\n"
      f"Answer: {results['answer']}")

# COMMAND ----------

query = "รณภูมิคู่ใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "ชีวาคู่ใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "ในท้ายที่่สุด พรชีวาคู่กับรณภูมิใช่ไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "ในท้ายที่่สุด พรชีวาคู่กับรณภูมิใช่ไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "พรชีวันคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "พรชีวันคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "พรชีวันคู่กับใคร"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "สรุจมีแฟนเก่าไหม"
results = rag_chain_with_source.invoke(query)

# COMMAND ----------

query = "ชีวากับดาลัด"
results = rag_chain_with_source.invoke(query)
