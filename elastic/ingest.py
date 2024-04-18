import argparse
import asyncio
import logging
import os
import sys
import nltk
nltk.download('averaged_perceptron_tagger')

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import llama_index.core
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore  # type: ignore
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    UnstructuredReader,
    FlatReader,
    HTMLTagReader,
)
from elasticsearch.helpers import BulkIndexError

from custom_llamaindex import CloudObjectStorageReader
from configs import AppConfig, IngestConfig, FileStoreConfig, FileStoreType
from elastic.utils import (
    get_async_elasticsearch_client_from_env,
    get_elasticsearch_client_from_env,
)
from elastic.setup import create_index_with_elser_ingestion_pipeline

logger = logging.getLogger(__name__)


def parse_main_args() -> str:
    """
    Parses the command line arguments and returns the path to the config file.

    Returns:
        str: The path to the config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        "-c",
        default="configs/ibm_config.yaml",
        help="Path to the config file",
    )
    return parser.parse_args().config_file_path


def get_file_reader_from_config(config: FileStoreConfig) -> BaseReader:
    """
    Returns a file reader based on the provided configuration.

    Args:
        config (FileStoreConfig): The configuration object containing information about the file store.

    Returns:
        BaseReader: An instance of the file reader based on the file store type.

    Raises:
        ValueError: If the file store type is unsupported.
    """
    DEFAULT_READERS = {
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".pptx": UnstructuredReader(),
        ".txt": FlatReader(),
        ".html": HTMLTagReader(),
    }
    num_files_limit = config.num_files_to_ingest or 5
    reader_kwargs = {
        "num_files_limit": num_files_limit,
        "file_extractor": DEFAULT_READERS,
    }
    file_store_type = FileStoreType(config.type)
    match file_store_type:
        case FileStoreType.COS:
            return CloudObjectStorageReader.from_service_credentials(
                bucket=config.location,
                service_credentials_path=config.service_credentials_path,
                num_files_limit=num_files_limit,
                file_extractor=DEFAULT_READERS,
            )
        case FileStoreType.LOCAL:
            return llama_index.core.SimpleDirectoryReader(
                input_dir=config.location, **reader_kwargs
            )
        case _:
            raise ValueError(f"Unsupported file store type: {file_store_type}")


def initialize_index_from_config(
    config: IngestConfig, show_progress=True, create_index_in_elastic=True
) -> llama_index.core.VectorStoreIndex:
    """
    Initializes and returns a VectorStoreIndex based on the provided configuration.
    The index uses an Elasticsearch index as its StorageContext.

    Args:
        config (IngestConfig): The configuration object containing the necessary settings.
        show_progress (bool, optional): Whether to show progress during the initialization process. Defaults to True.
        create_index_in_elastic (bool, optional): Whether to create the index in Elasticsearch. Defaults to True.

    Returns:
        llama_index.core.VectorStoreIndex: The initialized VectorStoreIndex.

    Raises:
        BulkIndexError: If there is an error during bulk indexing.

    """
    INDEX_NAME = config.elasticsearch_config.index_name
    INDEX_TEXT_FIELD = config.elasticsearch_config.index_text_field
    CHUNK_SIZE = config.chunk_size
    CHUNK_OVERLAP = config.chunk_overlap
    BATCH_SIZE = 50  # The amount of nodes Elasticsearch will ingest per bulk request

    file_storage = get_file_reader_from_config(config.file_store_config)
    documents = file_storage.load_data(show_progress=show_progress)

    # Embedding is handled by Elasticsearch instead of LlamaIndex
    llama_index.core.Settings.embed_model = None
    llama_index.core.Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    if create_index_in_elastic:
        es_client = get_elasticsearch_client_from_env()
        create_index_with_elser_ingestion_pipeline(
            es_client,
            index_name=INDEX_NAME,
            index_text_field=INDEX_TEXT_FIELD,
        )

    vector_store = ElasticsearchStore(
        es_client=get_async_elasticsearch_client_from_env(timeout=180),
        index_name=INDEX_NAME,
        text_field=INDEX_TEXT_FIELD,
        batch_size=BATCH_SIZE,
    )
    vector_store_add_kwargs = {
        "create_index_if_not_exists": False,
    }
    try:
        index = llama_index.core.VectorStoreIndex.from_documents(
            documents,
            storage_context=llama_index.core.StorageContext.from_defaults(
                vector_store=vector_store
            ),
            show_progress=show_progress,
            **vector_store_add_kwargs,
        )
    except BulkIndexError as e:
        logger.error("Bulk index error: %s", e)
        firstError = e.errors[0].get("index", {}).get("error", {})
        logger.error(f"First error reason: {firstError.get('reason')}")
        raise
    return index


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    config_file_path = parse_main_args()
    config = AppConfig.from_yaml(config_file_path).ingest
    index = initialize_index_from_config(config, show_progress=True)
    asyncio.run(index.vector_store.client.close())


if __name__ == "__main__":
    main()
