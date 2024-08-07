"""This file should be imported only and only if you want to run the UI locally."""
import datetime
import logging
import os.path
import sys
import threading
import time
from typing import Any, List, Literal
from gradio.queueing import Queue, Event
import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from gradio.blocks import BlockFunction
from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.ui.images import *
from private_gpt.ui.logging_custom import FileLogger

import re
import uuid
import tempfile
import pandas as pd
from tinydb import TinyDB, where
from llama_cpp import Llama
from private_gpt.paths import models_path
from huggingface_hub.file_download import http_get
from gradio_client.documentation import document
from private_gpt.templates.template import create_doc
from private_gpt.settings.settings import settings
import chromadb
import requests
from gradio_modal import Modal
from langchain_community.vectorstores import Chroma
from private_gpt.paths import local_data_path
from langchain_community.embeddings import HuggingFaceEmbeddings
from private_gpt.paths import models_cache_path
import socket
import qrcode.image.svg

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

logging_path = os.path.join(PROJECT_ROOT_PATH, "logging")
if not os.path.exists(logging_path):
    os.mkdir(logging_path)
f_logger = FileLogger(__name__, f"{logging_path}/answers_bot.log", mode='a', level=logging.INFO)

DATA_QUESTIONS = os.path.join(PROJECT_ROOT_PATH, "data_questions")
if not os.path.exists(DATA_QUESTIONS):
    os.mkdir(DATA_QUESTIONS)

IP_ADDRESS = f"{socket.gethostbyname(socket.gethostname())}:{settings().server.port}"
img = qrcode.make(IP_ADDRESS)
with open(QRCODE_PATH, 'wb') as qr:
    img.save(qr)


BLOCK_CSS = """

#buttons button {
    min-width: min(120px,100%);
}

/* Применяем стили для td */
tr focus {
    user-select: all; /* Разрешаем выделение текста */
}

/* Применяем стили для ячейки span внутри td */
tr span {
    user-select: all; /* Разрешаем выделение текста */
}

.message-bubble-border.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {
  border-style: none;
}

.user {
    background: #2042b9;
    color: white;
}

@media (min-width: 1024px) {
    .modal-container.svelte-7knbu5 {
        max-width: 50% !important
    }
}


.gap.svelte-1m1obck {
    padding: 4%
}

#login_btn {
    width: 250px;
    height: 40px;
}

"""

JS = """
function disable_btn() {
    var elements = document.getElementsByClassName('wrap default minimal svelte-1occ011 translucent');

    for (var i = 0; i < elements.length; i++) {
        if (elements[i].classList.contains('generating') || !elements[i].classList.contains('hide')) {
            // Выполнить любое действие здесь
            console.log('Элемент содержит класс generating');
            // Например:
            document.getElementById('component-35').disabled = true
            setTimeout(() => { document.getElementById('component-35').disabled = false }, 180000);
        }
    }
}
"""

LOCAL_STORAGE = """
function() {
    globalThis.setStorage = (key, value) => {
        localStorage.setItem(key, JSON.stringify(value))
    }
    globalThis.removeStorage = (key) => {
        localStorage.removeItem(key)
    }
    globalThis.getStorage = (key, value) => {
        return JSON.parse(localStorage.getItem(key))
    }
    const access_token = getStorage('access_token')
    return [access_token];
}
"""

UI_TAB_TITLE = "Ruscon AI"

SOURCES_SEPARATOR = "\n\n Документы: \n"

MODES = ["ВНД", "Свободное общение", "Получение документов"]
MAX_NEW_TOKENS: int = 1500
LINEBREAK_TOKEN: int = 13
SYSTEM_TOKEN: int = 1788
USER_TOKEN: int = 1404
BOT_TOKEN: int = 9225
CHUNK_SIZE: int = 1408
CHUNK_OVERLAP: int = 400

ROLE_TOKENS: dict = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


class Modes:
    DB = MODES[0]
    LLM = MODES[1]
    DOC = MODES[2]


class Blocks(gr.Blocks):
    from gradio.themes import ThemeClass as Theme

    def __init__(self,
                 theme: Theme | str | None = None,
                 analytics_enabled: bool | None = None,
                 mode: str = "blocks",
                 title: str = "Gradio",
                 css: str | None = None,
                 js: str | None = None,
                 head: str | None = None
                 ):
        super().__init__(theme, analytics_enabled, mode, title, css, js, head)
        self.app = None
        self.config = None
        self._queue = None

    @document()
    def queue(
            self,
            status_update_rate: float | Literal["auto"] = "auto",
            api_open: bool | None = None,
            max_size: int | None = None,
            concurrency_count: int | None = None,
            *,
            default_concurrency_limit: int | None | Literal["not_set"] = "not_set",
    ):
        from gradio import utils, routes

        if concurrency_count:
            raise DeprecationWarning(
                "concurrency_count has been deprecated. Set the concurrency_limit directly on event listeners "
                "e.g. btn.click(fn, ..., concurrency_limit=10) or gr.Interface(concurrency_limit=10). "
                "If necessary, the total number of workers can be configured via `max_threads` in launch()."
            )
        if api_open is not None:
            self.api_open = api_open
        if utils.is_zero_gpu_space():
            max_size = 1 if max_size is None else max_size
        self._queue = Queue(
            live_updates=status_update_rate == "auto",
            concurrency_count=self.max_threads,
            update_intervals=status_update_rate if status_update_rate != "auto" else 1,
            max_size=max_size,
            block_fns=self.fns,
            default_concurrency_limit=default_concurrency_limit,
        )
        self.config = self.get_config_file()
        self.app = routes.App.create_app(self)
        return self


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> set["Source"]:
        curated_sources = set()

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = os.path.basename(doc_metadata.get("file_name", "-")) if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.add(source)

        return curated_sources


@singleton
class PrivateGptUi:
    semaphore = threading.Semaphore()

    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        self._chat_service.llm = self.init_model()
        self._ingest_service.ingest_component.embedding_component = self.init_embedding()
        self._chat_service.index = self.init_db()

        # Cache the UI blocks
        self._ui_block = None
        self._queue = 0

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)
        self.tiny_db = TinyDB(f'{DATA_QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)

        self.auth_token = None

    @staticmethod
    def init_model():
        path = str(models_path / settings().local.llm_hf_model_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                http_get(
                    f"https://huggingface.co/{settings().local.llm_hf_repo_id}/resolve/main/"
                    f"{settings().local.llm_hf_model_file}",
                    f
                )

        return Llama(
            n_gpu_layers=43,
            model_path=path,
            n_ctx=settings().llm.context_window,
            n_parts=1
        )

    @staticmethod
    def init_embedding():
        return HuggingFaceEmbeddings(
            model_name=settings().local.embedding_hf_model_name,
            cache_folder=str(models_cache_path),
        )

    def init_db(self):
        client = chromadb.PersistentClient(path=str(local_data_path))
        return Chroma(
            client=client,
            collection_name=self._chat_service.collection,
            embedding_function=self._ingest_service.ingest_component.embedding_component,
        )

    def load_model(self, is_load_model: bool):
        """

        :return:
        """
        if is_load_model:
            logger.info("Loaded files")
            gr.Info("Сервер будет перезагружаться, обновите страницу")
            time.sleep(5)
            sys.exit(1)
        else:
            logger.info("Clear model")
            self._chat_service.llm.reset()
            self._chat_service.llm.set_cache(None)
            del self._chat_service.llm

    def get_current_model(self):
        return os.path.basename(self._chat_service.llm.model_path)

    def _get_context(self, history: list[list[str]], mode: str, limit, uid, *_: Any):
        match mode:
            case Modes.DB:
                content, scores = self._chat_service.retrieve(
                    history=history,
                    use_context=True,
                    limit=limit,
                    uid=uid
                )
                return content, mode, scores
            case Modes.LLM:
                content, scores = self._chat_service.retrieve(
                    history=history,
                    use_context=False,
                    uid=uid
                )
                return content, mode, scores
            case Modes.DOC:
                content, scores = self._chat_service.retrieve(
                    history=history,
                    use_context=False,
                    uid=uid
                )
                return content, mode, scores

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.DB:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.LLM:
                p = settings().ui.default_chat_system_prompt
            case Modes.DOC:
                p = settings().ui.default_chat_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self._system_prompt:
            return gr.update(placeholder=self._system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self._system_prompt, interactive=False)

    def _list_ingested_files(self):
        db = self._ingest_service.list_ingested_langchain()
        files = {
            os.path.basename(ingested_document["source"])
            for ingested_document in db["metadatas"]
        }
        return list(files)

    def user(self, message, history):
        uid = uuid.uuid4()
        logger.info(f"Обработка вопроса. Очередь - {self._queue}. UID - [{uid}]")
        self.semaphore.acquire()
        if history is None:
            history = []
        new_history = history + [[message, None]]
        self._queue += 1
        self.semaphore.release()
        logger.info(f"Закончена обработка вопроса. UID - [{uid}]")
        return "", new_history, uid

    def regenerate_response(self, history):
        """

        :param history:
        :return:
        """
        uid = uuid.uuid4()
        logger.info(f"Обработка вопроса. Очередь - {self._queue}. UID - [{uid}]")
        self.semaphore.acquire()
        self._queue += 1
        self.semaphore.release()
        logger.info(f"Закончена обработка вопроса. UID - [{uid}]")
        return "", history, uid

    def stop(self, uid):
        logger.info(f"Остановлено генерирование ответа. UID - [{uid}]")
        self.semaphore.release()

    def get_message_generator(self, history, retrieved_docs, mode, top_k, top_p, temp, uid):
        model = self._chat_service.llm
        last_user_message = history[-1][0]
        pattern = r'<a\s+[^>]*>(.*?)</a>'
        files = re.findall(pattern, retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        if retrieved_docs and mode == Modes.DB:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        elif mode == Modes.DOC:
            last_user_message = f"{last_user_message}\n\n" \
                                f"Сегодня {datetime.datetime.now().strftime('%d.%m.%Y')} число. " \
                                f"Если в контексте не указан год, то пиши {datetime.date.today().year}. " \
                                f"Напиши ответ только так, без каких либо дополнений: " \
                                f"Прошу предоставить ежегодный оплачиваемый отпуск с " \
                                f"(дата начала отпуска в формате DD.MM.YYYY) по " \
                                f"(дата окончания отпуска в формате DD.MM.YYYY)."
        logger.info(f"Вопрос был полностью сформирован [uid - {uid}]")
        f_logger.finfo(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Вопрос: {history[-1][0]} - "
                       f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")

        history_user = [
            {"role": "user", "content": user_message}
            for user_message, _ in history[-4:-1]
        ]
        generator = model.create_chat_completion(
            messages=[
                {
                    "role": "system", "content": self._system_prompt
                },
                *history_user,
                {
                    "role": "user", "content": last_user_message
                },

            ],
            stream=True,
            temperature=temp,
            top_k=top_k,
            top_p=top_p
        )
        return model, generator, files

    @staticmethod
    def calculate_end_date(history):
        long_days = re.findall(r"\d{1,4} д", history[-1][0])
        list_dates = []
        for day in long_days:
            day = int(day.replace(" д", ""))
            start_dates = re.findall(r"\d{1,2}[.]\d{1,2}[.]\d{2,4}", history[-1][1])
            for date in start_dates:
                list_dates.append(date)
                end_date = datetime.datetime.strptime(date, '%d.%m.%Y') + datetime.timedelta(days=day)
                end_date = end_date.strftime('%d.%m.%Y')
                list_dates.append(end_date)
                return [[f"Начало отпуска - {list_dates[0]}. Конец отпуска - {list_dates[1]}", None]]

    def get_dates_in_question(self, history, model, generator, mode):
        if mode == Modes.DOC:
            partial_text = ""
            for i, token in enumerate(generator):
                if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                    break
                letters = model.detokenize([token]).decode("utf-8", "ignore")
                partial_text += letters
                f_logger.finfo(letters)
                history[-1][1] = partial_text
            return self.calculate_end_date(history)

    @staticmethod
    def get_list_files(history, mode, scores, files, partial_text):
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = [
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            ]
            threshold = 0.34
            logger.info(f"Score is {scores[0]}")
            if scores and scores[0] <= threshold:
                partial_text += "\n\n\n".join(sources_text)
            elif scores:
                partial_text += f"\n\n⚠️ Похоже, данные в Базе знаний слабо соответствуют вашему запросу. " \
                                    f"Попробуйте подробнее описать ваш запрос или перейти в режим {MODES[1]}, " \
                                    f"чтобы общаться с ботом вне контекста Базы знаний"
            history[-1][1] = partial_text
        elif mode == Modes.DOC:
            file = create_doc(partial_text, "Титова", "Сергея Сергеевича", "Руководитель отдела",
                              "Отдел организационного развития")
            partial_text += f'\n\n\nФайл: {file}'
            history[-1][1] = partial_text
        return history

    def bot(self, history, retrieved_docs, mode, top_k, top_p, temp, uid, scores):
        """

        :param history:
        :param retrieved_docs:
        :param mode:
        :param top_k:
        :param top_p:
        :param temp:
        :param uid:
        :param scores:
        :return:
        """
        logger.info(f"Подготовка к генерации ответа. Формирование полного вопроса на основе контекста и истории "
                    f"[uid - {uid}]")
        self.semaphore.acquire()
        if not history or not history[-1][0]:
            yield history[:-1]
            self.semaphore.release()
            return
        model, generator, files = self.get_message_generator(history, retrieved_docs, mode, top_k, top_p, temp, uid)
        partial_text = ""
        logger.info(f"Начинается генерация ответа [uid - {uid}]")
        f_logger.finfo(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ответ: ")
        if message := self.get_dates_in_question(history, model, generator, mode):
            model, generator, files = self.get_message_generator(message, retrieved_docs, mode, top_k, top_p, temp, uid)
        elif mode == Modes.DOC:
            model, generator, files = self.get_message_generator(history, retrieved_docs, mode, top_k, top_p, temp, uid)
        try:
            token: dict
            for token in generator:
                for data in token["choices"]:
                    letters = data["delta"].get("content", "")
                    partial_text += letters
                    f_logger.finfo(letters)
                    history[-1][1] = partial_text
                    yield history
        except Exception as ex:
            logger.error(f"Error - {ex}")
            partial_text += "\nСлишком большой контекст. " \
                            "Попробуйте уменьшить его или измените количество выдаваемого контекста в настройках"
            history[-1][1] = partial_text
            yield history
            self.semaphore.release()
        f_logger.finfo(f" - [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
        logger.info(f"Генерация ответа закончена [uid - {uid}]")
        yield self.get_list_files(history, mode, scores, files, partial_text)
        self._queue -= 1
        self.semaphore.release()

    def _chat(self, history, context, mode, top_k, top_p, temp, uid, scores):
        match mode:
            case Modes.DB:
                yield from self.bot(history, context, Modes.DB, top_k, top_p, temp, uid, scores)
            case Modes.LLM:
                yield from self.bot(history, context, Modes.LLM, top_k, top_p, temp, uid, scores)
            case Modes.DOC:
                yield from self.bot(history, context, Modes.DOC, top_k, top_p, temp, uid, scores)

    def update_file(self, files: List[tempfile.TemporaryFile], chunk_size, chunk_overlap, uuid, uuid_old):
        db = self._ingest_service.list_ingested_langchain()
        pattern = re.compile(fr'{uuid_old}_\d*$')
        self._chat_service.index.delete([x for x in db["ids"] if pattern.match(x)])
        len_chunks = self._ingest_service.bulk_ingest(files, chunk_size, chunk_overlap, uuid)
        return f"Обновлено на {len_chunks} фрагментов! Можно задавать вопросы.", \
            gr.update(choices=self._list_ingested_files())

    def upload_file(self, files: List[tempfile.TemporaryFile], chunk_size: int, chunk_overlap: int, uuid: str = None):
        logger.debug("Loading count=%s files", len(files))
        len_chunks = self._ingest_service.bulk_ingest(files, chunk_size, chunk_overlap, uuid)
        return f"Загружено {len_chunks} фрагментов! Можно задавать вопросы.", \
            gr.update(choices=self._list_ingested_files())

    def delete_file(self, uuid):
        logger.info(f"UUID is {uuid}")
        db = self._ingest_service.list_ingested_langchain()

        for_delete_ids: list = [
            doc_id
            for ingested_document, doc_id in zip(db["metadatas"], db["ids"])
            if doc_id.rsplit("_", maxsplit=1)[0] in uuid or os.path.basename(ingested_document['source']) in uuid
        ]
        if for_delete_ids:
            self._ingest_service.delete(for_delete_ids)
        return f"Удалено {len(for_delete_ids)} фрагментов! Можно задавать вопросы.", \
            gr.update(choices=self._list_ingested_files())

    def get_analytics(self):
        try:
            return pd.DataFrame(self.tiny_db.all()).sort_values('Старт обработки запроса', ascending=False)
        except KeyError:
            return pd.DataFrame(self.tiny_db.all())

    @staticmethod
    def login(username, password):
        """

        :param username:
        :param password:
        :return:
        """
        response = requests.post(
            f"http://{IP_ADDRESS}/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code == 200:
            return {"access_token": response.json()["access_token"], "is_success": True}
        logger.error(response.json()["detail"])
        return {"access_token": None, "is_success": False, "message": response.json()["detail"]}

    def get_current_user_info(self, local_data, is_visible: bool = False):
        """

        :param local_data:
        :param is_visible:
        :return:
        """
        if isinstance(local_data, dict) and local_data.get("is_success", False):
            response = requests.get(
                f"http://{IP_ADDRESS}/users/me",
                headers={"Authorization": f"Bearer {local_data['access_token']}"}
            )
            logger.info(f"User is {response.json().get('username')}")
            is_logged_in = response.status_code == 200
            if is_logged_in:
                is_visible = False
        else:
            is_logged_in = False
        obj_tabs = [local_data] + [gr.update(visible=is_logged_in) for _ in range(3)]
        if is_logged_in:
            obj_tabs.append(gr.update(value="Выйти", icon=str(LOGOUT_ICON)))
        else:
            obj_tabs.append(gr.update(value="Войти", icon=str(LOGIN_ICON)))
        obj_tabs.append(gr.update(visible=is_visible))
        if isinstance(local_data, dict):
            obj_tabs.append(local_data.get("message", MESSAGE_LOGIN))
        else:
            obj_tabs.append(MESSAGE_LOGIN)
        obj_tabs.append(gr.update(choices=self._list_ingested_files()))
        return obj_tabs

    def login_or_logout(self, local_data, login_btn, is_visible):
        """

        :param local_data:
        :param login_btn:
        :param is_visible:
        :return:
        """
        data = self.get_current_user_info(local_data, is_visible=is_visible)
        if isinstance(data[0], dict) and data[0].get("access_token"):
            obj_tabs = [gr.update(visible=False)] + [gr.update(visible=False) for _ in range(3)]
            obj_tabs.append(gr.update(value="Войти", icon=str(LOGIN_ICON)))
            obj_tabs.append(gr.update(choices=self._list_ingested_files()))
            return obj_tabs
        obj_tabs = [gr.update(visible=True)] + [gr.update(visible=False) for _ in range(3)]
        obj_tabs.append(login_btn)
        obj_tabs.append(gr.update(choices=self._list_ingested_files()))
        return obj_tabs

    def calculate_analytics(self, messages, analyse=None):
        message = messages[-1][0] if messages else None
        answer = messages[-1][1] if message else None
        filter_query = where('Сообщения') == message
        if result := self.tiny_db.search(filter_query):
            if analyse is None:
                self.tiny_db.update(
                    {
                        'Ответы': answer,
                        'Количество повторений': result[0]['Количество повторений'] + 1,
                        'Старт обработки запроса': str(datetime.datetime.now())
                    },
                    cond=filter_query
                )
            else:
                self.tiny_db.update({'Оценка ответа': analyse}, cond=filter_query)
                gr.Info("Отзыв ответу поставлен")
        elif message is not None:
            self.tiny_db.insert(
                {'Сообщения': message, 'Ответы': answer, 'Количество повторений': 1, 'Оценка ответа': None,
                 'Старт обработки запроса': str(datetime.datetime.now())}
            )
        return self.get_analytics()

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.info("Creating the UI blocks")
        with Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft().set(
                body_background_fill="white",
                block_background_fill="#e1e5e8",
                block_label_background_fill="#2042b9",
                block_label_background_fill_dark="#2042b9",
                block_label_text_color="white",
                checkbox_label_background_fill_selected="#1f419b",
                checkbox_label_background_fill_selected_dark="#1f419b",
                checkbox_background_color_selected="#111d3d",
                checkbox_background_color_selected_dark="#111d3d",
                input_background_fill="#e1e5e8",
                button_primary_background_fill="#1f419b",
                button_primary_background_fill_dark="#1f419b",
                shadow_drop_lg="5px 5px 5px 5px rgb(0 0 0 / 0.1)"
            ),
            css=BLOCK_CSS,
            js=JS
        ) as blocks:
            # Ваш логотип и текст заголовка
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            header_html = f"""<h1><center>{logo_svg} Виртуальный ассистент Рускон (бета-версия)</center></h1>"""

            with gr.Row():
                gr.HTML(header_html)
                login_btn = gr.DuplicateButton("Войти", variant="primary", size="lg", elem_id="login_btn",
                                               icon=str(LOGIN_ICON))

            uid = gr.State(None)
            scores = gr.State(None)
            is_visible = gr.State(True)
            local_data = gr.JSON({}, visible=False)

            with gr.Tab("Чат"):
                with gr.Row():
                    mode = gr.Radio(
                        MODES[:2],
                        value=MODES[0],
                        show_label=False
                    )
                with gr.Row():
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot(
                            label="Диалог",
                            height=500,
                            show_copy_button=True,
                            avatar_images=(
                                AVATAR_USER,
                                AVATAR_BOT
                            )
                        )

                with gr.Row():
                    with gr.Column(scale=20):
                        msg = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            placeholder="👉 Напишите запрос",
                            container=False
                        )
                    with gr.Column(scale=3, min_width=100):
                        submit = gr.Button("📤 Отправить", variant="primary")

                with gr.Row(elem_id="buttons"):
                    like = gr.Button(value="👍 Понравилось")
                    dislike = gr.Button(value="👎 Не понравилось")
                    # stop = gr.Button(value="⛔ Остановить")
                    # regenerate = gr.Button(value="🔄 Повторить")
                    clear = gr.ClearButton(value="🗑️ Очистить")

                with gr.Row():
                    gr.HTML(
                        "<h5>"
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                        "</h5>"
                    )

            with gr.Tab("Документы", visible=False) as documents_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_button = gr.Files(
                            label="Загрузка документов",
                            file_count="multiple"
                        )
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")

                    with gr.Column(scale=7):
                        files_selected = gr.Dropdown(
                            choices=self._list_ingested_files(),
                            label="Выберите файлы для удаления",
                            value=None,
                            multiselect=True
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")

            with gr.Tab("Настройки", visible=False) as settings_tab:
                with gr.Row(elem_id="model_selector_row"):
                    models: list = [
                        f"{settings().local.llm_hf_repo_id.split('/')[1]}/{settings().local.llm_hf_model_file}"
                    ]
                    gr.Dropdown(
                        choices=models,
                        value=models[0],
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                with gr.Accordion("QR Code", open=False):
                    gr.Image(QRCODE_PATH, width=400, height=400)
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        limit = gr.Slider(
                            minimum=1,
                            maximum=12,
                            value=6,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=128,
                            maximum=1792,
                            value=CHUNK_SIZE,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=400,
                            value=CHUNK_OVERLAP,
                            step=10,
                            interactive=True,
                            label="Пересечение"
                        )
                    with gr.Tab(label="Параметры генерации"):
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        top_k = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=80,
                            step=5,
                            interactive=True,
                            label="Top-k",
                        )
                        temp = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.1,
                            step=0.1,
                            interactive=True,
                            label="Temp"
                        )

                with gr.Accordion("Системный промпт", open=False):
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        lines=5,
                        show_label=False
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                with gr.Accordion("Контекст", open=True):
                    with gr.Column(variant="compact"):
                        content = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

            with gr.Tab("Логи диалогов", visible=False) as logging_tab:
                with gr.Row():
                    with gr.Column():
                        analytics = gr.DataFrame(
                            value=self.get_analytics,
                            interactive=False,
                            wrap=True,
                            # column_widths=[200]
                        )

            with Modal(visible=False) as modal:
                with gr.Column(variant="panel"):
                    gr.HTML("<h1><center>Вход</center></h1>")
                    message_login = gr.HTML(MESSAGE_LOGIN)
                    login = gr.Textbox(
                        label="Логин",
                        placeholder="Введите логин",
                        show_label=True,
                        max_lines=1
                    )
                    password = gr.Textbox(
                        label="Пароль",
                        placeholder="Введите пароль",
                        show_label=True,
                        type="password"
                    )
                    submit_login = gr.Button("👤 Войти", variant="primary")
                    cancel_login = gr.Button("⛔ Отмена", variant="secondary")

            submit_login.click(
                fn=self.login,
                inputs=[login, password],
                outputs=[local_data]
            ).success(
                fn=self.get_current_user_info,
                inputs=[local_data, is_visible],
                outputs=[local_data, documents_tab, settings_tab, logging_tab, login_btn, modal, message_login,
                         files_selected]
            ).success(
                fn=None,
                inputs=[local_data],
                outputs=None,
                js="(v) => {setStorage('access_token', v)}"
            )

            login_btn.click(
                fn=self.login_or_logout,
                inputs=[local_data, login_btn, is_visible],
                outputs=[modal, documents_tab, settings_tab, logging_tab, login_btn, files_selected]
            ).success(
                fn=None,
                inputs=None,
                outputs=[local_data],
                js="() => {removeStorage('access_token')}"
            )
            cancel_login.click(
                fn=lambda: Modal(visible=False),
                inputs=None,
                outputs=modal
            )

            mode.change(
                self._set_current_mode, inputs=mode, outputs=system_prompt_input
            )

            upload_button.upload(
                self.upload_file,
                inputs=[upload_button, chunk_size, chunk_overlap],
                outputs=[file_warning, files_selected],
            )

            # Delete documents from db
            delete.click(
                fn=self.delete_file,
                inputs=files_selected,
                outputs=[file_warning, files_selected]
            )

            # Pressing Enter
            msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self._get_context,
                inputs=[chatbot, mode, limit, uid],
                outputs=[content, mode, scores],
                queue=True,
            ).success(
                fn=self._chat,
                inputs=[chatbot, content, mode, top_k, top_p, temp, uid, scores],
                outputs=[chatbot],
                queue=True,
            ).success(
                fn=self.calculate_analytics,
                inputs=chatbot,
                outputs=analytics,
                queue=True,
            )

            # Pressing the button
            submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self._get_context,
                inputs=[chatbot, mode, limit, uid],
                outputs=[content, mode, scores],
                queue=True,
            ).success(
                fn=self._chat,
                inputs=[chatbot, content, mode, top_k, top_p, temp, uid, scores],
                outputs=[chatbot],
                queue=True,
            ).success(
                fn=self.calculate_analytics,
                inputs=chatbot,
                outputs=analytics,
                queue=True,
            )

            # Like
            like.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, like],
                outputs=[analytics],
                queue=True,
            )

            # Dislike
            dislike.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, dislike],
                outputs=[analytics],
                queue=True,
            )

            # Clear history
            clear.click(
                fn=lambda: None,
                inputs=None,
                outputs=chatbot,
                queue=False,
                js=JS
            )
            # blocks.auth = self.login
            # blocks.auth_message = "Введите логин и пароль, чтобы войти"

            blocks.load(
                fn=self.get_current_user_info,
                inputs=[local_data],
                outputs=[local_data, documents_tab, settings_tab, logging_tab, login_btn, modal, message_login,
                         files_selected],
                js=LOCAL_STORAGE
            )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


class Queue(Queue):
    def __init__(
            self,
            live_updates: bool,
            concurrency_count: int,
            update_intervals: float,
            max_size: int | None,
            block_fns: list[BlockFunction],
            default_concurrency_limit: int | None | Literal["not_set"] = "not_set"
    ):
        super().__init__(live_updates, concurrency_count, update_intervals, max_size, block_fns,
                         default_concurrency_limit)

    def send_message(
            self,
            event: Event,
            message_type: str,
            data: dict | None = None,
    ):
        data = {} if data is None else data
        if messages := self.pending_messages_per_session.get(event.session_hash):
            messages.put_nowait({"msg": message_type, "event_id": event._id, **data})
        else:
            PrivateGptUi.semaphore.release()


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False, allowed_paths=["."])
