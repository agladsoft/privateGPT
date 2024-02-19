import os
import re
import logging
import subprocess
from pathlib import Path

from docx import Document as DocDocument
from langchain.schema import Document
# from llama_index import Document
from llama_index.readers import JSONReader, StringIterableReader
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from typing import Optional, List, Union, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import SpacyTextSplitter

logger = logging.getLogger(__name__)

# Patching the default file reader to support other file types
FILE_READER_CLS = DEFAULT_FILE_READER_CLS.copy()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)

LOADER_MAPPING: dict = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        def add_period_after_sentence(text, step=1024):
            for i in range(0, len(text), step):
                substring = text[i:i + step]
                if '.' not in substring and i + step < len(text):
                    # Find the last space in the substring and add a period after it
                    last_space = substring.rfind('\n')
                    if last_space != -1:
                        text = text[:i + last_space + 1].strip() + '.\n\n' + text[i + last_space + 1:]
            return text

        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = str(file_data)
            document.text = add_period_after_sentence(
                re.sub(r'(\s{2,}|\n{2,})', lambda match: match.group()[0]*2, document.text)
            )
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )

            if extension == ".doc":
                # Create a Document object
                subprocess.call(['soffice', '--headless', '--convert-to', 'docx', '--outdir',
                                 os.path.dirname("/".join(file_data.parts)[1:]),
                                 "/".join(file_data.parts)[1:]])

                doc = DocDocument("/".join(file_data.parts)[1:].replace(".doc", ".docx"))

                text = ''.join(paragraph.text + '\n' for paragraph in doc.paragraphs)
                string_reader = StringIterableReader()
                return string_reader.load_data([text])

            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        return reader_cls().load_data(file_data)

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]


class IngestionHelperLangchain:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        load_documents: List[Document], text_splitter: SpacyTextSplitter
    ) -> tuple[str, list[Document]]:
        def process_text(text: str) -> Optional[str]:
            """

            :param text:
            :return:
            """
            lines: list = text.split("\n")
            lines = [line for line in lines if len(line.strip()) > 2]
            text = "\n".join(lines).strip()
            return "" if len(text) < 10 else text

        documents = text_splitter.split_documents(load_documents)
        fixed_documents: List[Document] = []
        for doc in documents:
            doc.page_content = process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)
        return f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы.", fixed_documents

    @staticmethod
    def _load_file_to_documents(file_name: str) -> Document:
        logger.debug("Transforming file_name=%s into documents", file_name)
        ext: str = "." + file_name.rsplit(".", 1)[-1]
        assert ext in LOADER_MAPPING
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_name, **loader_args)
        logger.debug("Specific reader found for extension=%s", ext)
        document = loader.load()[0]
        document.page_content = re.sub(r'(\s{3,}|\n{3,})', lambda match: match.group()[0]*3,
                                       document.page_content)
        return document
