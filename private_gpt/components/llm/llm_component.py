import os
import logging
from injector import inject, singleton
from huggingface_hub.file_download import http_get
from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings

from llama_cpp import Llama

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    llm: Llama

    @inject
    def __init__(self, settings: Settings) -> None:
        llm_mode = settings.llm.mode
        logger.info("Initializing the LLM in mode=%s", llm_mode)
        match settings.llm.mode:
            case "local":
                os.makedirs(models_path, exist_ok=True)
                paths = [
                    str(models_path / os.path.basename(settings.local.llm_hf_repo_id[0]) /
                        settings.local.llm_hf_model_file[0]),
                    str(models_path / os.path.basename(settings.local.llm_hf_repo_id[0]) / settings.local.chat_format)
                ]
                for path in paths:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            http_get(
                                f"https://huggingface.co/{settings.local.llm_hf_repo_id[0]}/resolve/main/"
                                f"{os.path.basename(path)}",
                                f
                            )

                self.llm = None

            # case "sagemaker":
            #     from private_gpt.components.llm.custom.sagemaker import SagemakerLLM
            #
            #     self.llm = SagemakerLLM(
            #         endpoint_name=settings.sagemaker.llm_endpoint_name,
            #     )
            # case "openai":
            #     from llama_index.llms import OpenAI
            #
            #     openai_settings = settings.openai
            #     self.llm = OpenAI(
            #         api_key=openai_settings.api_key, model=openai_settings.model
            #     )
            # case "mock":
            #     self.llm = MockLLM()
