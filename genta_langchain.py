"""
Genta-Langchain Connector

This module provides classes for integrating Genta API with the Langchain package,
allowing developers to utilize Genta's AI inference solutions within their 
Langchain-based applications.

Classes:
    - GentaEmbeddings: Utilizes GentaEmbedding for Langchain usage.
    - GentaLLM: Utilizes Genta API LLMs for Langchain usage.
"""
from typing import Any, List, Dict, Mapping, Optional
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from requests.exceptions import JSONDecodeError
from pydantic import Field
from genta import GentaAPI


class GentaEmbeddings(Embeddings):
    """
    GentaEmbeddings class for utilizing GentaEmbedding in Langchain.

    This class inherits from the Embeddings base class and provides methods for
    embedding documents and queries using the GentaEmbedding model.

    Attributes:
        API (GentaAPI): GentaAPI instance for making API calls.
        embedding_model_name (str): Name of the embedding model to use.
    """
    api: GentaAPI = Field(..., description="GentaAPI instance")
    embedding_model_name: str = Field(default="GentaEmbedding",
                                      description="Name of the model to use")

    def __init__(self, api, model_name) -> list:
        self.api = api
        self.embedding_model_name = model_name

    def embed_documents(self, texts) -> list:
        """
        Embed a list of documents using the GentaEmbedding model.

        Args:
            texts (List[str]): List of documents to embed.

        Returns:
            List[List[float]]: List of embeddings for each document.
        """
        embeddings = []
        for text in texts:
            try:
                embedding, _ = self.api.Embedding(
                    text=text, model_name=self.embedding_model_name)
                # Append the embedding to the list
                embeddings.append(embedding[0])
            except JSONDecodeError as error:
                print(f"Error decoding JSON response for text: {text}")
                print(f"Error message: {str(error)}")
                # Handle the error, e.g., skip the embedding or use a default value
                # Append None as a placeholder for the failed embedding
                embeddings.append(None)
        return embeddings  # Return the list of embeddings

    def embed_query(self, text):
        """
        Embed a single query using the GentaEmbedding model.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding for the query.
        """
        embedding, _ = self.api.Embedding(
            text=text, model_name="GentaEmbedding")
        return embedding[0]


class GentaLLM(LLM):
    """
    GentaLLM class for utilizing Genta API LLMs in Langchain.

    This class inherits from the LLM base class and provides methods for
    generating text using Genta API LLMs.

    Attributes:
        api (GentaAPI): GentaAPI instance for making API calls.
        model_name (str): Name of the LLM to use.    
    """
    api: GentaAPI = Field(..., description="GentaAPI instance")
    model_name: str = Field(default="Llama2-7B",
                            description="Name of the model to use")

    def _call(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 1024,
        stop: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = 1.03,
        temperature: Optional[float] = 0.7,
    ) -> str:
        """
        Generate text using the specified Genta API LLM.

        Args:
            prompt ([str]): Prompt asked to the Genta API
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate. 
                Default is 1024.
            stop (Optional[List[str]]): Optional list of stop sequences.
            repetition_penalty (Optional[float]): Penalty for repeated tokens. 
                Default is 1.03.
            temperature (Optional[float]): Sampling temperature. 
                Default is 0.7.
            top_p (Optional[float]): Cumulative probability threshold for top-p sampling. 
                Default is 0.95.
        Returns:
            str: Generated text.
        """

        response = self.api.ChatCompletion(
            chat_history=[{'role': 'user', 'content':prompt}],
            model_name=self.model_name,
            max_new_tokens = max_new_tokens,
            stop=stop,
            repetition_penalty=repetition_penalty,
            temperature=temperature
        )
        generated_text = response[0][0][0]['generated_text']
        return generated_text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "Genta"
