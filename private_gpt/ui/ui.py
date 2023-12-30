"""This file should be imported only and only if you want to run the UI locally."""
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import pandas as pd

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import FAVICON_PATH

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_USER = THIS_DIRECTORY_RELATIVE / "icons8-человек-96.png"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "icons8-bot-96.png"

UI_TAB_TITLE = "RusconGPT"

SOURCES_SEPARATOR = "\n\n Sources: \n"

MODES = ["DB", "Search in DB", "LLM"]

BLOCK_CSS = """

#buttons button {
    min-width: min(120px,100%);
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

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
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

    def _get_context(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                # Remove from history content the Sources information
                                content=interaction[1].split(SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:3]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        logger.info(f"Messages {all_messages}")
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case "DB":
                query_stream, content = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                )
                return "", history + [[message, None]], query_stream, content

            case "LLM":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                return "", history + [[message, None]], llm_stream, "Появятся после задавания вопросов"

            case "Search in DB":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)
                return "", history + [[message, None]], sources, "Появятся после задавания вопросов"

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

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def delete_doc(self, documents: str):
        logger.info(f"Documents is {documents}")
        return "", self._list_ingested_files()

    @staticmethod
    def regenerate_response(history):
        """

        :param history:
        :return:
        """
        return "", history

    @staticmethod
    def _chat(history, gen_message, mode):

        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                history[-1][1] = full_response
                yield history

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n".join(
                    f"{index}. {source.file} (page {source.page})"
                    for index, source in enumerate(cur_sources, start=1)
                )
                full_response += sources_text
            yield history

        match mode:
            case "DB":
                yield from yield_deltas(gen_message)
            case "LLM":
                yield from yield_deltas(gen_message)
            case "Search in DB":
                history[-1][1] = "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(gen_message, start=1)
                )
                yield history

    def _upload_file(self, files: list[str], ingested_dataset):
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]
        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])
        return "Файлы загружены", ingested_dataset

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(),
            css=BLOCK_CSS
        ) as blocks:
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{logo_svg} Я, Макар - текстовый ассистент на основе GPT</center></h1>"""
            )

            with gr.Tab("Чат"):
                response: gr.State = gr.State(None)

                with gr.Accordion("Контекст", open=False):
                    with gr.Column(variant="compact"):
                        content = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

                with gr.Row():
                    with gr.Column(scale=5, variant="compact"):
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
                        system_prompt_input = gr.Textbox(
                            placeholder=self._system_prompt,
                            label="Системный промпт",
                            lines=2,
                            interactive=True,
                            render=False,
                        )
                        mode.change(
                            fn=lambda c: c,
                            inputs=[mode]
                        )
                        # On blur, set system prompt to use in queries
                        system_prompt_input.blur(
                            self._set_system_prompt,
                            inputs=system_prompt_input,
                        )

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

            upload_button.upload(
                self._upload_file,
                inputs=[upload_button, ingested_dataset],
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
                fn=self._get_context,
                inputs=[msg, chatbot, mode],
                outputs=[msg, chatbot, response, content],
                queue=True,
            ).success(
                fn=self._chat,
                inputs=[chatbot, response, mode],
                outputs=chatbot,
                queue=True,
            )

            # Pressing the button
            submit_click_event = submit.click(
                fn=self._get_context,
                inputs=[msg, chatbot, mode],
                outputs=[msg, chatbot, response, content],
                queue=True,
            ).success(
                fn=self._chat,
                inputs=[chatbot, response, mode],
                outputs=chatbot,
                queue=True,
            )

            # Stop generation
            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event],
                queue=False,
            )

            # Regenerate
            regenerate.click(
                fn=self.regenerate_response,
                inputs=[chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self._get_context,
                inputs=[msg, chatbot, mode],
                outputs=[msg, chatbot, response, content],
                queue=True,
            ).success(
                fn=self._chat,
                inputs=[chatbot, response, mode],
                outputs=chatbot,
                queue=True,
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
