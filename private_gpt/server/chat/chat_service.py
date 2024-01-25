import os
import logging
from typing import Tuple, Union
from dataclasses import dataclass

from private_gpt.constants import PROJECT_ROOT_PATH

from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms import ChatMessage, MessageRole
from llama_index.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.embedding.embedding_component import EmbeddingComponent, EmbeddingComponentLangchain
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk

import chromadb
from langchain.vectorstores import Chroma
from private_gpt.paths import local_data_path
from langchain.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

FILES_DIR = os.path.join(PROJECT_ROOT_PATH, "upload_files")
os.makedirs(FILES_DIR, exist_ok=True)
os.chmod(FILES_DIR, 0o0777)
os.environ['GRADIO_TEMP_DIR'] = FILES_DIR


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponentLangchain,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm = llm_component.llm
        self.collection = "all-documents"
        client = chromadb.PersistentClient(path=str(local_data_path))
        self.index: Chroma = Chroma(
            client=client,
            collection_name=self.collection,
            embedding_function=embedding_component.embedding_model,
        )

    def retrieve(
        self,
        history,
        use_context: bool = False,
        limit: int = 2,
        uid: str = None
    ) -> str:
        if not use_context or not history or not history[-1][0]:
            return "Появятся после задавания вопросов"
        last_user_message = history[-1][0]
        docs = self.index.similarity_search_with_score(last_user_message, limit)
        data: dict = {}
        for doc in docs:
            url = f"""<a href="file/{doc[0].metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{os.path.basename(doc[0].metadata["source"])}</a>"""
            document: str = f'Документ - {url} ↓'
            if document in data:
                data[document] += "\n\n" + f"Score: {round(doc[1], 2)}, Text: {doc[0].page_content}"
            else:
                data[document] = f"Score: {round(doc[1], 2)}, Text: {doc[0].page_content}"
        list_data: list = [f"{doc}\n\n{text}" for doc, text in data.items()]
        logger.info(f"Получили контекст из базы [uid - {uid}]")
        return "\n\n\n".join(list_data) if list_data else "Документов в базе нету"

    def _chat_engine(
        self,
        limit: int,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None
    ) -> BaseChatEngine:
        if use_context:
            vector_index_retriever = self.vector_store_component.get_retriever(
                index=self.index, context_filter=context_filter, similarity_top_k=limit
            )
            context_template = (
                "Контекстная информация приведена ниже."
                "\n--------------------\n"
                "{context_str}"
                "\n--------------------\n"
            )
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=vector_index_retriever,
                service_context=self.service_context,
                context_template=context_template,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window"),
                ],
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                service_context=self.service_context,
            )

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion
