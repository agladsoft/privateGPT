import os
import logging
from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM
from huggingface_hub.file_download import http_get
from private_gpt.components.llm.prompt_helper import get_prompt_style
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
                path = str(models_path / settings.local.llm_hf_model_file)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        http_get(
                            f"https://huggingface.co/{settings.local.llm_hf_repo_id}/resolve/main/"
                            f"{settings.local.llm_hf_model_file}",
                            f
                        )

                self.llm = Llama(
                    n_gpu_layers=35,
                    model_path=path,
                    n_ctx=settings.llm.context_window,
                    n_parts=1
                )

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
