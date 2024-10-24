import os
import re
import magic
import logging
import sqlite3
import subprocess
from pathlib import Path
from docx import Document as DocDocument
from deep_translator import GoogleTranslator
from llama_index.readers import JSONReader, StringIterableReader
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

from langchain_community.document_loaders import (
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
    UnstructuredExcelLoader
)
from typing import Optional, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from private_gpt.paths import local_data_path

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
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
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

MIME_TYPE: dict = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".png": "image/png"
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
        load_documents: List[Document], text_splitter: RecursiveCharacterTextSplitter
    ) -> tuple[int, list[Document]]:
        def process_text(text: str) -> Optional[str]:
            """

            :param text:
            :return:
            """
            lines: list = text.split("\n")
            lines = [line for line in lines if len(line.strip()) > 2]
            text = "\n".join(lines).strip()
            return "" if len(text) < 10 else text

        fixed_documents: List[Document] = []

        for doc in load_documents:
            # Проверка на формат Excel по расширению файла
            if doc.metadata.get("source", "").endswith(".xlsx"):
                # Не разделяем документ, просто добавляем его в список как есть
                fixed_documents.append(doc)
                continue

            # Разделяем остальные документы с помощью text_splitter
            split_docs = text_splitter.split_documents([doc])
            for split_doc in split_docs:
                split_doc.page_content = process_text(split_doc.page_content)
                if split_doc.page_content:
                    fixed_documents.append(split_doc)

        return len(fixed_documents), fixed_documents

    @staticmethod
    def _load_file_to_documents(file_name: str) -> Document:
        def remove_time(date_str):
            if isinstance(date_str, str):
                # Поиск даты в формате ГГГГ-ММ-ДД с последующим временем
                return re.sub(r'\s*00:00:00$', '', date_str)
            return date_str

        def get_extension(file_name_: str) -> str:
            mime_type: str = magic.Magic(mime=True).from_file(file_name_)
            for ext_, mime in MIME_TYPE.items():
                if mime_type == mime:
                    return ext_
            return ""

        def clean_column_name(translator_, col_name):
            translated_col = translator_.translate(col_name)

            # Приведение к нижнему регистру, удаление скобок и замена проблемных символов на подчеркивания
            cleaned_col = re.sub(r'[^\w\s]', '', translated_col).replace(' ', '_').lower()
            return cleaned_col

        logger.debug("Transforming file_name=%s into documents", file_name)
        file_name_without_ext, ext = os.path.splitext(file_name)
        if ext == "" or ext not in MIME_TYPE:
            ext = get_extension(file_name)
        assert ext in LOADER_MAPPING, f"Не поддерживается формат {ext}"
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_name, **loader_args)
        logger.debug("Specific reader found for extension=%s", ext)
        try:
            document = loader.load()[0]
        except OSError as ex:
            logger.error(f"Exception is {ex}. Type of {type(ex)}")
            raise BrokenPipeError("Загружен битый файл")
        dict_formats = {
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".csv": pd.read_csv
        }
        if ext in dict_formats:
            translator = GoogleTranslator(source='ru', target='en')
            df = dict_formats[ext](file_name, dtype=str, keep_default_na=False)
            df.columns = [clean_column_name(translator, col) for col in df.columns]
            df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
            df = df.map(remove_time)

            conn = sqlite3.connect(f'{local_data_path}/users.db')
            df.to_sql(
                clean_column_name(translator, os.path.basename(file_name).replace(ext, "")),
                conn,
                if_exists='replace',
                index=False
            )
            conn.close()

            result_str = "\n\n".join(
                "\n".join(f"{header}: {row[header]}" for header in df.columns)
                for _, row in df.iterrows()
            )
            document.page_content = result_str.strip()
        else:
            document.page_content = re.sub(r'(\s{3,}|\n{3,})', lambda match: match.group()[0]*3, document.page_content)
        return document
