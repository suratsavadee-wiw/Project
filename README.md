# Intern Project
Creating a chatbot to answer questions about the Thai novel using RAG techniques.

**Overview**

Sometimes, we need to read or search for information in a 40-50 page document, but we don't have enough time to go through the entire document. This project aims to solve that problem by creating a chatbot that can search, summarize, and answer questions about a long document.

In this example, the document is from a Thai novel called "Pornchiwan" (พรชีวัน), which has over 500 pages. Using Retrieval-Augmented Generation (RAG), we can improve the chatbot’s accuracy while reducing hallucinations, making it a more reliable tool for answering questions about the novel.

**How It Works**

The chatbot uses Large Language Models (LLMs) combined with RAG techniques to retrieve relevant information and generate human-like

**What is RAG?**

RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by:

1. Retrieving relevant content from external sources (e.g., vector databases).

2. Augmenting the retrieved data into the prompt.

3. Generating responses based on both the query and the retrieved context.

This approach prevents hallucinations and ensures responses are accurate and contextually relevant.

**Project Pipeline**

1. Indexing Process

- Extract text: Convert the novel’s PDF into text using OCR (Optical Character Recognition).

- Preprocess & clean: Remove unnecessary characters and format the text.

- Chunking: Split the text into smaller sections for efficient search and retrieval.

- Embedding & storage: Convert text into vector embeddings and store them in a vector database.

2. Retrieval & Generation

- User query: A user inputs a question about the novel.

- Retrieve relevant context: The system finds the most relevant chunks of text based on cosine similarity.

- Generate response: The chatbot combines the retrieved text with the user’s query and generates an answer using an LLM.

**Here's my sevice workflow :**

<img width="581" alt="Screenshot 2568-03-03 at 15 51 06" src="https://github.com/user-attachments/assets/bf2c741e-c9be-4b53-8597-19044752d3ff" />

**Demo: "Novel App" with sreamlit implement**

<img width="493" alt="Screenshot 2568-03-03 at 15 52 55" src="https://github.com/user-attachments/assets/2c48a86d-69a5-46d1-be8f-af8a701a24aa" />
