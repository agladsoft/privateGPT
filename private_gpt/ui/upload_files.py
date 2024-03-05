from langchain.embeddings import HuggingFaceEmbeddings
import sys
# from private_gpt.components.embedding import embedding_component
from private_gpt.components.ingest.ingest_component import BaseIngestComponentLangchain, \
    get_ingestion_component_langchain
from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import settings


embedding_component = HuggingFaceEmbeddings(
    model_name=settings().local.embedding_hf_model_name,
    cache_folder=str(models_cache_path),
)


ingest_component: BaseIngestComponentLangchain = \
    get_ingestion_component_langchain(embedding_component, settings=settings())


ingest_component.bulk_ingest(
    sys.argv[1:],
    1408,
    400
)
