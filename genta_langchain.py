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
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic import Field
from genta import GentaAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult


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
    model_name: str = Field(default="Llama2-7B", alias='model_name')
    api_token: str

    def __init__(self, api_token: str, model_name: str = "Llama2-7B", **kwargs):
        super().__init__(api_token=api_token, **kwargs)
        self.model_name = model_name
        self.api_token = api_token
        self._lc_kwargs = kwargs

    @property
    def api(self) -> GentaAPI:
        """
        Returns an instance of the GentaAPI class with the provided API token.

        :return: An instance of the GentaAPI class.
        :rtype: GentaAPI
        """
        return GentaAPI(token=self.api_token)

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
        return response[0]['generated_text']

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


class GentaChatLLM(BaseChatModel):
    """
    GentaChatLLM is a class that represents a Genta Chat Language Model.

    Args:
        api_token (str): The API token for accessing the GentaAPI.
        model_name (str, optional): The name of the language model. Defaults to "Llama2-7B".
        **kwargs: Additional keyword arguments.

    Attributes:
        model_name (str): The name of the language model.
        api_token (str): The API token for accessing the GentaAPI.
        _lc_kwargs (dict): Additional keyword arguments.

    """

    model_name: str = Field(default="Llama2-7B", alias='model_name')
    api_token: str
    temperature: float = Field(default=1.1, alias='model_name')
    max_new_token: int = Field(default=256, alias='model_name')

    def __init__(self, api_token: str, model_name: str = "Llama2-7B", temperature: Optional[float] = 1.1, max_new_token: Optional[int] = 256, **kwargs):
        super().__init__(api_token=api_token, **kwargs)
        self.model_name = model_name
        self.api_token = api_token
        self._lc_kwargs = kwargs
        self.temperature = temperature
        self.max_new_token = max_new_token

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generates a response based on the given messages.

        Args:
            messages (List[BaseMessage]): The list of messages in the conversation.
            stop (List[str], optional): List of stop tokens to stop the generation. Defaults to None.
            run_manager (CallbackManagerForLLMRun, optional): Callback manager for LLM run. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult: The generated response.

        """

        response, _ = self.api.ChatCompletion(
            chat_history=self.format_messages(messages),
            model_name=self.model_name,
            stop=stop,
            max_new_tokens=self.max_new_token,
            temperature=self.temperature,
            **self._lc_kwargs
        )

        return ChatResult(
            generations=[ChatGeneration(
                message=AIMessage(
                    content=response[0][0]['generated_text']
                )
            )]
        )

    def format_messages(self, messages: list) -> list:
        """
        Formats the messages into the required format for chat completion.

        Args:
            messages (list): The list of messages to be formatted.

        Returns:
            list: The formatted messages.

        """
        reformatted_message = []
        for message in messages:
            if isinstance(message, SystemMessage):
                reformatted_message.append(
                    {"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                reformatted_message.append(
                    {"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                reformatted_message.append(
                    {"role": "assistant", "content": message.content})
        return reformatted_message

    @property
    def api(self) -> GentaAPI:
        """
        Returns an instance of the GentaAPI class with the provided API token.

        Returns:
            GentaAPI: An instance of the GentaAPI class.

        """
        return GentaAPI(token=self.api_token)

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
        return "Genta Chat"
