import logging
import tempfile
from pathlib import Path
from typing import BinaryIO

from injector import inject, singleton
from llama_index import (
    ServiceContext,
    StorageContext,
)
from llama_index.node_parser import SentenceWindowNodeParser

from private_gpt.components.embedding.embedding_component import EmbeddingComponentLangchain
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings

import os
import pandas as pd
from typing import Union, List
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from private_gpt.components.ingest.ingest_component import get_ingestion_component_langchain, BaseIngestComponentLangchain


logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponentLangchain,
        node_store_component: NodeStoreComponent
    ) -> None:
        # self.llm_service = llm_component
        # self.storage_context = StorageContext.from_defaults(
        #     vector_store=vector_store_component.vector_store,
        #     docstore=node_store_component.doc_store,
        #     index_store=node_store_component.index_store,
        # )
        # node_parser = SentenceWindowNodeParser.from_defaults()
        # self.ingest_service_context = ServiceContext.from_defaults(
        #     llm=self.llm_service.llm,
        #     embed_model=embedding_component.embedding_model,
        #     node_parser=node_parser,
        #     # Embeddings done early in the pipeline of node transformations, right
        #     # after the node parsing
        #     transformations=[node_parser, embedding_component.embedding_model],
        # )

        # self.ingest_component = get_ingestion_component_langchain(
        #     self.storage_context, self.ingest_service_context, embedding_component, settings=settings()
        # )

        self.ingest_component: BaseIngestComponentLangchain = \
            get_ingestion_component_langchain(embedding_component, settings=settings())

    def ingest(self, file_name: str, file_data: Path) -> Union[str, list[IngestedDoc]]:
        logger.info("Ingesting file_name=%s", file_name)
        message, documents = self.ingest_component.ingest(file_name, file_data)
        return message, [IngestedDoc.from_document(document) for document in documents]

    def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def bulk_ingest(self, files: List[str], chunk_size: int, chunk_overlap: int):
        logger.debug("Ingesting file_names=%s", [f for f in files])
        return self.ingest_component.bulk_ingest(files, chunk_size, chunk_overlap)

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs = []
        try:
            docstore = self.storage_context.docstore
            ingested_docs_ids: set[str] = set()

            for node in docstore.docs.values():
                if node.ref_doc_id is not None:
                    ingested_docs_ids.add(node.ref_doc_id)

            for doc_id in ingested_docs_ids:
                ref_doc_info = docstore.get_ref_doc_info(ref_doc_id=doc_id)
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def list_ingested_langchain(self):
        return self.ingest_component._index.get()

    def delete(self, doc_ids: list) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_ids
        )
        self.ingest_component._index.delete(doc_ids)
