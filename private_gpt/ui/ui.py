"""This file should be imported only and only if you want to run the UI locally."""
import logging
import os.path
from pathlib import Path
from typing import Any, List

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import FAVICON_PATH

import re
import uuid
import tempfile
import pandas as pd

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_USER = THIS_DIRECTORY_RELATIVE / "icons8-человек-96.png"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "icons8-bot-96.png"

UI_TAB_TITLE = "MakarGPT"

SOURCES_SEPARATOR = "\n\n Документы: \n"

MODES = ["DB", "LLM"]
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

"""


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

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)

    def _get_context(self, history: list[list[str]], mode: str, limit, uid, *_: Any):
        match mode:
            case "DB":
                content, scores = self._chat_service.retrieve(
                    history=history,
                    use_context=True,
                    limit=limit,
                    uid=uid
                )
                return content, mode, scores
            case "LLM":
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
            case "DB":
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case "LLM":
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

        # for node in self._ingest_service.list_ingested_langchain():
        #     if node.doc_id is not None and os.path.basename(node.doc_metadata["file_name"]) in list_documents:
        #         self._ingest_service.delete(node.doc_id)
        # return "", self._list_ingested_files()

    @staticmethod
    def user(message, history):
        uid = uuid.uuid4()
        logger.info(f"Обработка вопроса [uid - {uid}]")
        if history is None:
            history = []
        new_history = history + [[message, None]]
        return "", new_history, uid

    @staticmethod
    def regenerate_response(history):
        """

        :param history:
        :return:
        """
        uid = uuid.uuid4()
        logger.info(f"Обработка вопроса [uid - {uid}]")
        return "", history, uid

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
        for i, token in enumerate(generator):
            if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                break
            partial_text += model.detokenize([token]).decode("utf-8", "ignore")
            history[-1][1] = partial_text
            yield history
        logger.info(f"Генерация ответа закончена [uid - {uid}]")
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = "\n\n\n".join(
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            )
            partial_text += sources_text
            if scores and scores[0] > 4:
                partial_text += f"\n\n⚠️ Похоже, данные в Базе знаний слабо соответствуют вашему запросу. " \
                                f"Попробуйте подробнее описать ваш запрос или перейти в режим {MODES[1]}, " \
                                f"чтобы общаться с Макаром вне контекста Базы знаний"
            history[-1][1] = partial_text
        yield history

    def _chat(self, history, context, mode, uid, scores):
        match mode:
            case "DB":
                yield from self.bot(history, context, True, uid, scores)
            case "LLM":
                yield from self.bot(history, context, False, uid, scores)

    def _upload_file(self, files: List[tempfile.TemporaryFile], chunk_size: int, chunk_overlap: int):
        logger.debug("Loading count=%s files", len(files))
        message = self._ingest_service.bulk_ingest([f.name for f in files], chunk_size, chunk_overlap)
        return message, self._list_ingested_files()

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(),
            css=BLOCK_CSS
        ) as blocks:
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{logo_svg} Я, Макар - виртуальный ассистент Рускон</center></h1>"""
            )
            uid = gr.State(None)
            scores = gr.State(None)

            with gr.Tab("Чат"):
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        limit = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=4,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=50,
                            maximum=1024,
                            value=1024,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=100,
                            step=10,
                            interactive=True,
                            label="Пересечение"
                        )

                with gr.Accordion("Контекст", open=False):
                    with gr.Column(variant="compact"):
                        content = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

                with gr.Row():
                    with gr.Column(scale=4, variant="compact"):
                        mode = gr.Radio(
                            MODES,
                            label="Коллекции",
                            value="DB",
                            info="Переключение между выбором коллекций. Нужен ли контекст или нет?"
                        )
                        upload_button = gr.Files(
                            file_count="multiple"
                        )
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")

                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot(
                            label="Диалог",
                            height=500,
                            show_copy_button=True,
                            show_share_button=True,
                            avatar_images=(
                                AVATAR_USER,
                                AVATAR_BOT
                            )
                        )
                        with gr.Accordion("Системный промпт", open=False):
                            system_prompt_input = gr.Textbox(
                                placeholder=self._system_prompt,
                                label="Системный промпт",
                                lines=2
                            )
                            # On blur, set system prompt to use in queries
                            system_prompt_input.blur(
                                self._set_system_prompt,
                                inputs=system_prompt_input,
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
                    gr.Button(value="👍 Понравилось")
                    gr.Button(value="👎 Не понравилось")
                    stop = gr.Button(value="⛔ Остановить")
                    regenerate = gr.Button(value="🔄 Повторить")
                    clear = gr.Button(value="🗑️ Очистить")

                with gr.Row():
                    gr.Markdown(
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                    )

            with gr.Tab("Документы"):
                with gr.Row():
                    with gr.Column(scale=3):
                        find_doc = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
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
            submit_event = msg.submit(
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
                outputs=chatbot,
                queue=True,
            )

            # Pressing the button
            submit_click_event = submit.click(
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
                outputs=chatbot,
                queue=True,
            )

            # Regenerate
            regenerate_click_event = regenerate.click(
                fn=self.regenerate_response,
                inputs=[chatbot],
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
                outputs=chatbot,
                queue=True,
            )

            # Stop generation
            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event, regenerate_click_event],
                queue=False,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False)

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
    _blocks.launch(debug=False, show_api=False)
