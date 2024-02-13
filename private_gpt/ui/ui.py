"""This file should be imported only and only if you want to run the UI locally."""
import datetime
import logging
import os.path
import threading
from pathlib import Path
from typing import Any, List

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import FAVICON_PATH
from private_gpt.ui.logging_custom import FileLogger

import re
import uuid
import tempfile
import pandas as pd
from tinydb import TinyDB, where
from private_gpt.templates.template import create_doc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

logging_path = os.path.join(PROJECT_ROOT_PATH, "logging")
if not os.path.exists(logging_path):
    os.mkdir(logging_path)
f_logger = FileLogger(__name__, f"{logging_path}/answers_bot.log", mode='a', level=logging.INFO)

DATA_QUESTIONS = os.path.join(PROJECT_ROOT_PATH, "data_questions")
if not os.path.exists(DATA_QUESTIONS):
    os.mkdir(DATA_QUESTIONS)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_USER = THIS_DIRECTORY_RELATIVE / "icons8-человек-96.png"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "icons8-bot-96.png"
js = """
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

UI_TAB_TITLE = "MakarGPT"

SOURCES_SEPARATOR = "\n\n Документы: \n"

MODES = ["ВНД", "Свободное общение", "Получение документов"]
MAX_NEW_TOKENS: int = 1500
LINEBREAK_TOKEN: int = 13
SYSTEM_TOKEN: int = 1788
USER_TOKEN: int = 1404
BOT_TOKEN: int = 9225

ROLE_TOKENS: dict = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

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

.message-buttons-user {
  border-style: none;
}

.message-buttons-bot {
  border-style: none;
}

"""

JS = """
function checkClassExists() {
    if (!document.body.classList.contains("dark")) {
        document.querySelector(".svelte-90oupt").style.background = "#e1e5e8";
        document.querySelector(".svelte-nab2ao").style.background = "#e1e5e8";
        document.querySelector(".svelte-1ed2p3z").style.background = "#ffffff";
    }
}
"""


class Modes:
    DB = MODES[0]
    LLM = MODES[1]
    DOC = MODES[2]


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

        # Cache the UI blocks
        self._ui_block = None
        self._queue = 0

        self.semaphore = threading.Semaphore()

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)
        self.tiny_db = TinyDB(f'{DATA_QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)

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
        p = ""
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
        return pd.DataFrame({"Название файлов": list(files)})

    def delete_doc(self, documents: str):
        logger.info(f"Documents is {documents}")
        for_delete_ids: list = []
        list_documents: list[str] = documents.strip().split("\n")
        db = self._ingest_service.list_ingested_langchain()

        for ingested_document, doc_id in zip(db["metadatas"], db["ids"]):
            print(ingested_document)
            if os.path.basename(ingested_document["source"]) in list_documents:
                for_delete_ids.append(doc_id)
        if for_delete_ids:
            self._ingest_service.delete(for_delete_ids)
        return "", self._list_ingested_files()

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

    @staticmethod
    def get_message_tokens(model, role: str, content: str) -> list:
        """

        :param model:
        :param role:
        :param content:
        :return:
        """
        message_tokens: list = model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, ROLE_TOKENS[role])
        message_tokens.insert(2, LINEBREAK_TOKEN)
        message_tokens.append(model.token_eos())
        return message_tokens

    def get_system_tokens(self, model) -> list:
        """

        :param model:
        :return:
        """
        system_message: dict = {"role": "system", "content": self._system_prompt}
        return self.get_message_tokens(model, **system_message)

    def bot(self, history, retrieved_docs, mode, uid, scores):
        """

        :param history:
        :param retrieved_docs:
        :param mode:
        :param uid:
        :param scores:
        :return:
        """
        logger.info(f"Подготовка к генерации ответа. Формирование полного вопроса на основе контекста и истории "
                    f"[uid - {uid}]")
        self.semaphore.acquire()
        if not history or not history[-1][0]:
            yield history[:-1]
            return
        model = self._chat_service.llm
        tokens = self.get_system_tokens(model)[:]
        tokens.append(LINEBREAK_TOKEN)

        for user_message, bot_message in history[-4:-1]:
            message_tokens = self.get_message_tokens(model=model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]
        pattern = r'<a\s+[^>]*>(.*?)</a>'
        files = re.findall(pattern, retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        if retrieved_docs and mode:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        message_tokens = self.get_message_tokens(model=model, role="user", content=last_user_message)
        tokens.extend(message_tokens)
        logger.info(f"Вопрос был полностью сформирован [uid - {uid}]")
        f_logger.finfo(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Вопрос: {history[-1][0]} - "
                       f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")

        role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = model.generate(
            tokens,
            top_k=80,
            top_p=0.9,
            temp=0.1
        )

        partial_text = ""
        logger.info(f"Начинается генерация ответа [uid - {uid}]")
        f_logger.finfo(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ответ: ")
        try:
            for i, token in enumerate(generator):
                if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                    break
                letters = model.detokenize([token]).decode("utf-8", "ignore")
                partial_text += letters
                f_logger.finfo(letters)
                history[-1][1] = partial_text
                yield history
        except Exception as ex:
            logger.error(f"Error - {ex}")
            partial_text += "Слишком большой контекст. " \
                            "Попробуйте уменьшить его или измените количество выдаваемого контекста в настройках"
            history[-1][1] = partial_text
            yield history
            self.semaphore.release()
        f_logger.finfo(f" - [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
        logger.info(f"Генерация ответа закончена [uid - {uid}]")
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = [
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            ]
            if scores and scores[0] < 4:
                partial_text += "\n\n\n".join(sources_text)
            elif scores and scores[0] > 4:
                partial_text += sources_text[0]
            history[-1][1] = partial_text
        yield history
        self._queue -= 1
        self.semaphore.release()

    def get_documents(self, history, uid):
        logger.info(f"Подготовка к генерации ответа. Формирование полного вопроса на основе контекста и истории "
                    f"[uid - {uid}]")
        self.semaphore.acquire()
        if not history or not history[-1][0]:
            yield history[:-1]
            return
        model = self._chat_service.llm
        tokens = self.get_system_tokens(model)[:]
        tokens.append(LINEBREAK_TOKEN)

        for user_message, bot_message in history[-4:-1]:
            message_tokens = self.get_message_tokens(model=model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]

        last_user_message = f"{last_user_message}\n\n" \
                            f"Напиши ответ только так, без каких либо дополнений: " \
                            f"Прошу предоставить ежегодный оплачиваемый отпуск с " \
                            f"(дата начала отпуска в формате '%d.%m.%Y') по " \
                            f"(дата окончания отпуска в формате '%d.%m.%Y')."

        message_tokens = self.get_message_tokens(model=model, role="user", content=last_user_message)
        tokens.extend(message_tokens)
        role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = model.generate(
            tokens,
            top_k=80,
            top_p=0.9,
            temp=0.1
        )

        partial_text = ""
        logger.info(f"Начинается генерация ответа [uid - {uid}]")
        f_logger.finfo(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ответ: ")
        for i, token in enumerate(generator):
            if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                break
            letters = model.detokenize([token]).decode("utf-8", "ignore")
            partial_text += letters
            f_logger.finfo(letters)
            history[-1][1] = partial_text
            yield history
        f_logger.finfo(f" - [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
        logger.info(f"Генерация ответа закончена [uid - {uid}]")

        file = create_doc(partial_text, "Титова", "Сергея Сергеевича", "Руководитель отдела",
                          "Отдел организационного развития")
        partial_text += f'\n\n\nФайл: {file}'
        history[-1][1] = partial_text
        yield history
        self._queue -= 1
        self.semaphore.release()

    def _chat(self, history, context, mode, uid, scores):
        match mode:
            case Modes.DB:
                yield from self.bot(history, context, True, uid, scores)
            case Modes.LLM:
                yield from self.bot(history, context, False, uid, scores)
            case Modes.DOC:
                yield from self.get_documents(history, uid)

    def _upload_file(self, files: List[tempfile.TemporaryFile], chunk_size: int, chunk_overlap: int):
        logger.debug("Loading count=%s files", len(files))
        message = self._ingest_service.bulk_ingest([f.name for f in files], chunk_size, chunk_overlap)
        return message, self._list_ingested_files()

    def get_analytics(self):
        try:
            return pd.DataFrame(self.tiny_db.all()).sort_values('Старт обработки запроса', ascending=False)
        except KeyError:
            return pd.DataFrame(self.tiny_db.all())

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
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft().set(
                body_background_fill="white",
                block_label_background_fill="#2042b9",
                block_label_text_color="white",
                checkbox_label_background_fill_selected="#1f419b",
                input_background_fill="#e1e5e8",
                button_primary_background_fill="#1f419b"
            ),
            css=BLOCK_CSS,
            js=JS
        ) as blocks:
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{logo_svg} Я, Макар - виртуальный ассистент Рускон</center></h1>"""
            )
            uid = gr.State(None)
            scores = gr.State(None)

            with gr.Tab("Чат"):
                with gr.Row():
                    mode = gr.Radio(
                        MODES,
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
                    clear = gr.Button(value="🗑️ Очистить")

                with gr.Row():
                    gr.Markdown(
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                    )

            with gr.Tab("Документы"):
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_button = gr.Files(
                            label="Загрузка документов",
                            file_count="multiple"
                        )
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")
                        find_doc = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            info=" Напишите названия файлов, которые нужно удалить из базы ",
                            placeholder="👉 Напишите название документа",
                            container=False
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")
                    with gr.Column(scale=7):
                        ingested_dataset = gr.List(
                            self._list_ingested_files,
                            headers=["Название файлов"],
                            interactive=False,
                            render=False,  # Rendered under the button
                        )
                        ingested_dataset.change(
                            self._list_ingested_files,
                            outputs=ingested_dataset,
                        )
                        ingested_dataset.render()

            with gr.Tab("Настройки"):
                with gr.Row(elem_id="model_selector_row"):
                    models: list = list([f"{settings().local.llm_hf_repo_id.split('/')[1]}/"
                                         f"{settings().local.llm_hf_model_file}"])
                    gr.Dropdown(
                        choices=models,
                        value=models[0],
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        limit = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=50,
                            maximum=1536,
                            value=1200,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=300,
                            step=10,
                            interactive=True,
                            label="Пересечение"
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

            with gr.Tab("Логи диалогов"):
                with gr.Row():
                    with gr.Column():
                        analytics = gr.DataFrame(
                            value=self.get_analytics,
                            interactive=False,
                            wrap=True,
                            # column_widths=[200]
                        )

            mode.change(
                self._set_current_mode, inputs=mode, outputs=system_prompt_input
            )

            upload_button.upload(
                self._upload_file,
                inputs=[upload_button, chunk_size, chunk_overlap],
                outputs=[file_warning, ingested_dataset],
            )

            # Delete documents from db
            delete.click(
                fn=self.delete_doc,
                inputs=find_doc,
                outputs=[find_doc, ingested_dataset]
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
                inputs=[chatbot, content, mode, uid, scores],
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
                inputs=[chatbot, content, mode, uid, scores],
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
                js=js
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


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False, allowed_paths=["."])
