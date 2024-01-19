import logging

from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM

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
                from llama_index.llms import LlamaCPP

                prompt_style = get_prompt_style(settings.local.prompt_style)

                # self.llm = LlamaCPP(
                #     model_path=str(models_path / settings.local.llm_hf_model_file),
                #     temperature=0.1,
                #     max_new_tokens=settings.llm.max_new_tokens,
                #     # llama2 has a context window of 4096 tokens,
                #     # but we set it lower to allow for some wiggle room
                #     context_window=settings.llm.context_window,
                #     generate_kwargs={},
                #     # All to GPU
                #     model_kwargs={"n_gpu_layers": -1},
                #     # transform inputs into Llama2 format
                #     messages_to_prompt=prompt_style.messages_to_prompt,
                #     completion_to_prompt=prompt_style.completion_to_prompt,
                #     verbose=True,
                # )
                self.llm = Llama(
                    n_gpu_layers=35,
                    model_path=str(models_path / settings.local.llm_hf_model_file),
                    n_ctx=settings.llm.context_window,
                    n_parts=1,
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
