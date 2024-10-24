import logging

from injector import inject, singleton
from private_gpt.settings.settings import Settings

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model: HuggingFaceEmbeddings

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        match embedding_mode:
            case "local":
                self.embedding_model = None
            # case "sagemaker":
            #
            #     from private_gpt.components.embedding.custom.sagemaker import (
            #         SagemakerEmbedding,
            #     )
            #
            #     self.embedding_model = SagemakerEmbedding(
            #         endpoint_name=settings.sagemaker.embedding_endpoint_name,
            #     )
            # case "openai":
            #     from llama_index import OpenAIEmbedding
            #
            #     openai_settings = settings.openai.api_key
            #     self.embedding_model = OpenAIEmbedding(api_key=openai_settings)
            # case "mock":
            #     # Not a random number, is the dimensionality used by
            #     # the default embedding model
            #     self.embedding_model = MockEmbedding(128)


@singleton
class EmbeddingComponentLangchain:
    embedding_model: HuggingFaceEmbeddings

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        match embedding_mode:
            case "local":

                self.embedding_model = None
            # case "sagemaker":
            #
            #     from private_gpt.components.embedding.custom.sagemaker import (
            #         SagemakerEmbedding,
            #     )
            #
            #     self.embedding_model = SagemakerEmbedding(
            #         endpoint_name=settings.sagemaker.embedding_endpoint_name,
            #     )
            # case "openai":
            #     from llama_index import OpenAIEmbedding
            #
            #     openai_settings = settings.openai.api_key
            #     self.embedding_model = OpenAIEmbedding(api_key=openai_settings)
            # case "mock":
            #     # Not a random number, is the dimensionality used by
            #     # the default embedding model
            #     self.embedding_model = MockEmbedding(128)
