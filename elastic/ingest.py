import argparse
import asyncio
import logging
import json
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

import elasticsearch
import elastic_transport

from configs import AppConfig

from elasticsearch.helpers import BulkIndexError

from custom_llamaindex import CloudObjectStorageReader
from configs import AppConfig, IngestConfig, FileStoreConfig, FileStoreType
from elastic.utils import (
    replace_placeholders_in_JSON_template,
    get_async_elasticsearch_client_from_env,
    get_elasticsearch_client_from_env,
)

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

def create_index(
    client: elasticsearch.Elasticsearch,
    index_name: str,
    index_settings: dict,
    delete_existing: bool = False,
) -> elastic_transport.ObjectApiResponse:
    """
    Create an index in elasticsearch.Elasticsearch.

    Args:
        client (elasticsearch.Elasticsearch): The elasticsearch.Elasticsearch client.
        index_name (str): The name of the index to be created.
        index_settings (dict): The settings for the index.
        delete_existing (bool, optional): Whether to delete the existing index with the same name. Defaults to True.

    Returns:
        elastic_transport.ObjectApiResponse: The response from elasticsearch.Elasticsearch.

    """
    logger.info("Creating the index...")
    index_already_exists = client.indices.exists(index=index_name)
    if index_already_exists:
        if delete_existing:
            # Delete the existing index if requested
            client.indices.delete(index=index_name)
            # Create a new index after deletion
            response = client.indices.create(index=index_name, body=index_settings)
        else:
            # Return the existing index status
            response = client.indices.get(index=index_name)
    else:
        # Create the new index since it doesn't exist
        response = client.indices.create(index=index_name, body=index_settings)

    return response


def create_index_with_elser_ingestion_pipeline(
    client: elasticsearch.Elasticsearch,
    index_name: str,
    pipeline_name: str = "elser_ingestion_pipeline",
    index_text_field: str = "text",
    index_embedding_field: str = "sparse_embedding",
    embedding_model_id: str = ".elser_model_2_linux-x86_64",
) -> tuple[elastic_transport.ObjectApiResponse, elastic_transport.ObjectApiResponse]:
    """
    Creates an elasticsearch.Elasticsearch index that embeds docuemnt text into an embedding field
    using the ELSER model automatically on document ingestion.

    Args:
        client (elasticsearch.Elasticsearch): The elasticsearch.Elasticsearch client.
        index_name (str): The name of the index to be created.
        pipeline_name (str, optional): The name of the ingestion pipeline.
            Defaults to "elser_ingestion_pipeline".
        index_text_field (str, optional): The name of the field in the index containing the document text.
            Defaults to "text".
        index_embedding_field (str, optional): The name of the field in the index to store the document embeddings.
            Defaults to "sparse_embedding".
        embedding_model_id (str, optional): The ID of the model to use for text embedding.
            Defaults to ELSER i.e. "elser_model_2_linux-x86_64".

    Returns:
        tuple: A tuple containing the index creation response and the pipeline creation response.
    """
    response = client.ml.get_trained_models(model_id=embedding_model_id)
    embedding_model_text_field = response.body["trained_model_configs"][0]["input"]["field_names"][0]

    # Prepare pipeline config but replacing values in template
    with open("elastic/elastic_templates/pipeline_template.json") as f:
        elser_pipeline_template = json.load(f)

    pipeline_config = replace_placeholders_in_JSON_template(
        elser_pipeline_template,
        embedding_model_id=embedding_model_id,
        embedding_model_text_field=embedding_model_text_field,
        index_text_field=index_text_field,
        index_embedding_field=index_embedding_field,
    )

    pipeline_response = client.ingest.put_pipeline(
        id=pipeline_name, body=pipeline_config
    )

    # Prepare index config but replacing values in template
    with open("elastic/elastic_templates/index_template.json") as f:
        index_template = json.load(f)

    index_config = replace_placeholders_in_JSON_template(
        index_template,
        index_text_field=index_text_field,
        index_embedding_field=index_embedding_field,
    )

    index_config["settings"] = {
        "index.default_pipeline": pipeline_name,
    }

    index_response = create_index(
        client, index_name, index_config, delete_existing=False
    )

    return index_response, pipeline_response

def initialize_index_from_config(
    config: IngestConfig, show_progress=True
) -> llama_index.core.VectorStoreIndex:
    """
    Initializes and returns a VectorStoreIndex based on the provided configuration.
    The index uses an Elasticsearch index as its StorageContext.

    Args:
        config (IngestConfig): The configuration object containing the necessary settings.
        show_progress (bool, optional): Whether to show progress during the initialization process. Defaults to True.

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

    # Get the ES Client and create index and pipeline
    es_client = get_elasticsearch_client_from_env()
    create_index_with_elser_ingestion_pipeline(
        es_client,
        index_name=INDEX_NAME,
        index_text_field=INDEX_TEXT_FIELD,
    )
    
    # Create a Vector Store object with an async version of the ES client
    vector_store = ElasticsearchStore(
        es_client=get_async_elasticsearch_client_from_env(timeout=180),
        index_name=INDEX_NAME,
        text_field=INDEX_TEXT_FIELD,
        batch_size=BATCH_SIZE,
    )
    vector_store_add_kwargs = {
        "create_index_if_not_exists": True,
    }

    # Index the documents
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
