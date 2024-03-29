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
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import AIMessage, HumanMessage, SystemMessage
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
    GentaLLM is a class that represents a Genta Language Model.

    Args:
        api_token (str): The API token for accessing the GentaAPI.
        model_name (str, optional): The name of the Genta model to use. Defaults to "Llama2-7B".
        **kwargs: Additional keyword arguments.

    Attributes:
        model_name (str): The name of the Genta model.
        api (GentaAPI): An instance of the GentaAPI class.
        _lc_kwargs (dict): Additional keyword arguments for the GentaAPI.TextCompletion method.
    """

    model_name: str = Field(None, alias='model_name')
    api: GentaAPI

    def __init__(self, api_token:str, model_name:str = "Llama2-7B", **kwargs):
        super().__init__()
        self.model_name = model_name
        self.api = GentaAPI(token=api_token)
        self._lc_kwargs = kwargs

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:
        """
        Calls the GentaAPI.TextCompletion method to generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            stop (List[str], optional): A list of stop sequences to stop text generation. Defaults to None.
            run_manager (CallbackManagerForLLMRun, optional): A callback manager for monitoring the text generation process. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated text.

        """
        response, _ = self.api.TextCompletion(
            text=prompt,
            model_name=self.model_name,
            stop=stop,
            **self._lc_kwargs
        )
        return response[0][0][0]['generated_text']
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Returns the identifying parameters of the GentaLLM instance.

        Returns:
            Mapping[str, Any]: A dictionary of identifying parameters.
        """
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the language model.

        Returns:
            str: The type of the language model.
        """
        return "Genta"
