"""
This module contains example uses for the GentaLLM and GentaEmbedding class.
It demonstrates how to create an instance of GentaLLM and Genta Embedding
and generate text and genta embedding using the Genta API.
"""
from langchain.schema import HumanMessage, SystemMessage
from genta import GentaAPI
from genta_langchain import GentaEmbeddings, GentaLLM, GentaChatLLM

# Initialize GentaEmbeddings and GentaLLM
genta_api = GentaAPI("GENTA_API_TOKEN")

genta_llm = GentaLLM(api_token="GENTA_API_TOKEN")
genta_embeddings = GentaEmbeddings(api=genta_api, model_name="GentaEmbedding")

# Use GentaEmbeddings in Langchain
input_text = "Halo semua, ini Genta"
embeddings_text = genta_embeddings.embed_query(input_text)

input_texts = ["Genta Technology untuk kemajuan bersama", "Genta API mudah digunakan"]
embeddings_documents = genta_embeddings.embed_documents(input_texts)

# Use GentaLLM in Langchain (Text/prompt completion)
result = genta_llm.invoke("Apa merek permen karet yang bagus?")

print(embeddings_text)
print(embeddings_documents)
print(result)

# Use GentaChatLLM in Langchain (Chat completion)
genta_chat_llm = GentaChatLLM(api_token="GENTA_API_TOKEN", model_name="GentaEmbedding")

messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
]

result = genta_chat_llm.invoke(messages)