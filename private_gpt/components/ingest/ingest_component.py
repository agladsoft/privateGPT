import abc
import logging
import threading
from typing import Any

from llama_index import (
    Document,
)

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

import os
import chromadb
from typing import Union, List
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from private_gpt.components.ingest.ingest_helper import IngestionHelperLangchain
from private_gpt.components.embedding.embedding_component import EmbeddingComponentLangchain

logger = logging.getLogger(__name__)


class BaseIngestComponentLangchain(abc.ABC):
    def __init__(
        self,
        embedding_component: EmbeddingComponentLangchain,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.embedding_component = embedding_component

    @abc.abstractmethod
    def ingest(self, file_name: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: List[str], chunk_size: int, chunk_overlap: int, uuid) -> list[Document]:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass


class BaseIngestComponentWithIndexLangchain(BaseIngestComponentLangchain, abc.ABC):
    def __init__(
        self,
        embedding_component: EmbeddingComponentLangchain,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(embedding_component, *args, **kwargs)

        self.show_progress = True
        self._index_thread_lock = (
            threading.Lock()
        )  # Thread lock! Not Multiprocessing lock
        self.collection = "all-documents"
        try:
            embedding = embedding_component.embedding_model
        except AttributeError:
            embedding = embedding_component
        self._index: Chroma = self._initialize_index(embedding)

    def _initialize_index(self, embedding) -> Chroma:
        """Initialize the index from the storage context."""
        # Load the index with store_nodes_override=True to be able to delete them
        client = chromadb.PersistentClient(path=str(local_data_path))
        index: Chroma = Chroma(
            client=client,
            collection_name=self.collection,
            embedding_function=embedding
        )
        return index

    def _save_index(self) -> None:
        self._index.storage_context.persist(persist_dir=local_data_path)

    def delete(self, doc_id: str) -> None:
        with self._index_thread_lock:
            # Delete the document from the index
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

            # Save the index
            self._save_index()


class SimpleIngestComponentLangchain(BaseIngestComponentWithIndexLangchain):
    def __init__(
        self,
        embedding_component: EmbeddingComponentLangchain,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(embedding_component, *args, **kwargs)

    def ingest(self, file_name: str, chunk_size: int, chunk_overlap: int) -> Union[int, list[Document]]:
        logger.info("Ingesting file_name=%s", file_name)
        load_documents: List[Document] = [
            IngestionHelperLangchain._load_file_to_documents(path) for path in [file_name]
        ]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        message, documents = IngestionHelperLangchain.transform_file_into_documents(load_documents, text_splitter)
        ids: List[str] = [
            f"{os.path.basename(doc.metadata['source']).replace('.txt', '')}{i}"
            for i, doc in enumerate(documents)
        ]
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        self._save_docs(documents, ids)
        logger.info("Saving the documents in the index and doc store")
        return message

    def bulk_ingest(self, files: str, chunk_size: int, chunk_overlap: int, uuid) -> int:
        load_documents: List[Document] = [
            IngestionHelperLangchain._load_file_to_documents(path) for path in files
        ]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        message, documents = IngestionHelperLangchain.transform_file_into_documents(load_documents, text_splitter)
        if uuid:
            ids: List[str] = [f"{uuid}_{i}" for i, doc in enumerate(documents)]
        else:
            ids: List[str] = [f"{os.path.basename(doc.metadata['source']).replace('.txt', '')}_{i}"
                              for i, doc in enumerate(documents)]
        self._save_docs(documents, ids)
        return message

    def _save_docs(self, documents: list[Document], ids: List[str]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        self._index.from_documents(
            documents=documents,
            embedding=self.embedding_component,
            ids=ids,
            persist_directory=str(local_data_path),
            collection_name=self.collection,
        )
        logger.info("Persisting the index and nodes")
        return documents


def get_ingestion_component_langchain(
    embedding_component: EmbeddingComponentLangchain,
    settings: Settings,
) -> BaseIngestComponentLangchain:
    """Get the ingestion component for the given configuration."""
    # ingest_mode = settings.embedding.ingest_mode
    # if ingest_mode == "batch":
    #     return BatchIngestComponent(
    #         storage_context, service_context, settings.embedding.count_workers
    #     )
    # elif ingest_mode == "parallel":
    #     return ParallelizedIngestComponent(
    #         storage_context, service_context, settings.embedding.count_workers
    #     )
    # else:
    return SimpleIngestComponentLangchain(embedding_component)
