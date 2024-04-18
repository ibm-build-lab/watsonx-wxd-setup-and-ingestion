import asyncio
import dataclasses
import logging
import json
from typing import Dict, Union, Callable
import argparse
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import dotenv
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BaseEvaluator,
)

from utils import (
    get_async_elasticsearch_client_from_env,
    get_elasticsearch_client_from_env,
    replace_placeholders_in_JSON_template,
)
from configs import AppConfig, LLMConfig, LLMService
from custom_llamaindex import IbmBamLLM, CustomWatsonX


def create_sparse_vector_query_with_model(
    model_id: str, embedding_field: str = "sparse_embedding"
) -> Callable[[Dict, VectorStoreQuery], Dict]:
    """
    Creates a sparse vector query function with a specified model.
    Intended to be passed as a custom_query to the ElasticsearchStore query method.

    Args:
        model_id (str): The ID of the model to be used for text expansion.
        model_text_field (str, optional): The field in the model that contains the text.
            Defaults to "ml.tokens".

    Returns:
        Callable[[Dict, VectorStoreQuery], Dict]: A function that takes an existing query
            and a VectorStoreQuery object, and returns a new query with text expansion
            using the specified model.
    """

    def sparse_vector_query(existing_query: Dict, query: VectorStoreQuery) -> Dict:
        new_query = existing_query.copy()
        if query.mode in [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID]:
            new_query["query"] = {
                "text_expansion": {
                    f"ml.{embedding_field}": {
                        "model_id": model_id,
                        "model_text": query.query_str,
                    }
                }
            }
        return new_query

    return sparse_vector_query


def get_llm(config: LLMConfig, env_path=None) -> Union[CustomWatsonX, IbmBamLLM]:
    """
    Retrieves the appropriate LLM (Language Model) based on the provided configuration.

    Args:
        config (LLMConfig): The configuration object containing the necessary parameters for LLM retrieval.
        env_path (str, optional): The path to the environment file for authenticating to LLM service. Defaults to None.

    Returns:
        Union[CustomWatsonX, IbmBamLLM]: The LLM instance based on the provided configuration.

    Raises:
        EnvironmentError: If required environment variables are not set.
        ValueError: If the specified service is not supported.
    """

    dotenv.load_dotenv(dotenv_path=env_path)

    match config.service_name:
        case LLMService.BAM:
            apikey = os.environ.get("IBM_APIKEY")
            if apikey is None:
                raise EnvironmentError(
                    "IBM_APIKEY must be set in the environment if using BAM."
                )
            llm = IbmBamLLM(
                apikey=os.environ.get("GENAI_KEY"),
                model_id=config.llm_id,
                additional_kwargs=config.llm_params,
            )
        case LLMService.WML:
            apikey = os.environ.get("IBM_APIKEY")
            url = os.environ.get("WML_URL")
            project_id = os.environ.get("PROJECT_ID")
            if any([apikey is None, url is None, project_id is None]):
                raise EnvironmentError(
                    "IBM APIKEY, WML URL, and PROJECT_ID must be set in the environment if using WML."
                )
            llm = CustomWatsonX(
                credentials={
                    "apikey": os.environ.get("IBM_APIKEY"),
                    "url": os.environ.get("WML_URL"),
                },
                project_id=os.environ.get("PROJECT_ID"),
                model_id=config.llm_id,
                additional_kwargs=config.llm_params,
            )
        case _:
            raise ValueError(f"Service {config.service_name} is not supported.")
    return llm


def create_query_engine(
    config: AppConfig, index: VectorStoreIndex
) -> RetrieverQueryEngine:
    """
    Creates a retriever query engine based on the provided configuration and index.

    Args:
        config (AppConfig): The application configuration.
        index (VectorStoreIndex): The vector store index.

    Returns:
        RetrieverQueryEngine: The retriever query engine.

    Raises:
        FileNotFoundError: If the prompt template file specified in the configuration is not found.
    """
    NUM_DOCS_TO_RETRIEVE = config.query.num_docs_to_retrieve or 3
    PROMPT_TEMPLATE_PATH = config.query.prompt_template_path
    EMBEDDING_MODEL_ID = config.ingest.elasticsearch_config.embedding_model_id
    INDEX_EMBEDDING_FIELD = config.ingest.elasticsearch_config.index_embedding_field

    with open(PROMPT_TEMPLATE_PATH, "r") as file:
        prompt_template = file.read()
    retriever = VectorIndexRetriever(
        index=index,
        vector_store_query_mode="sparse",
        similarity_top_k=NUM_DOCS_TO_RETRIEVE,
        vector_store_kwargs={
            "custom_query": create_sparse_vector_query_with_model(
                EMBEDDING_MODEL_ID, INDEX_EMBEDDING_FIELD
            )
        },
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, text_qa_template=PromptTemplate(prompt_template)
    )
    return query_engine


@dataclasses.dataclass
class QueryScriptArgs:
    """
    Represents the arguments for the main function when running this file as ascript

    Args:
        config_file_path (str): The path to the configuration file.
        query (str): The query string.
        synthesize_response (bool, optional): Whether to synthesize the response. Defaults to True.
        use_evaluators (bool, optional): Whether to use evaluators. Defaults to True.
    """

    config_file_path: str
    query: str
    synthesize_response: bool = True
    use_evaluators: bool = True


def parse_main_args() -> QueryScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        "-c",
        default="configs/ibm_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--query", "-q", default=None, help="Query to search for in Elastic"
    )
    parser.add_argument(
        "--synthesize_response",
        "-s",
        action="store_true",
        help="Whether to synthesize a response from the query using Watsonx.ai",
    )
    parser.add_argument(
        "--use_evaluators",
        "-e",
        action="store_true",
        help="Whether to use LLM-based evaluators to evaluate the response",
    )
    args = parser.parse_args()
    return QueryScriptArgs(
        config_file_path=args.config_file_path,
        query=args.query,
        synthesize_response=args.synthesize_response,
        use_evaluators=args.use_evaluators,
    )


async def main():
    """
    Entry point of the program.

    This function performs the main logic of the program, including loading configuration,
    querying the Elasticsearch index, and synthesizing a RAG response with BAM and evaluating the response if specified.

    Args:
        None

    Returns:
        None
    """
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)

    args = parse_main_args()
    config = AppConfig.from_yaml(args.config_file_path)
    INDEX_NAME = config.ingest.elasticsearch_config.index_name
    TEXT_FIELD = config.ingest.elasticsearch_config.index_text_field
    Settings.embed_model = None

    if args.synthesize_response:
        LLM_PATH = config.query.llm_path
        Settings.llm = get_llm(LLMConfig.from_json_path(LLM_PATH), env_path=".env")
    else:
        Settings.llm = None

    client = get_async_elasticsearch_client_from_env()
    vector_store = ElasticsearchStore(
        index_name=INDEX_NAME,
        es_client=client,
        text_field=TEXT_FIELD,
    )
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = create_query_engine(config, vector_store_index)

    num_docs_in_index = (await client.count(index=INDEX_NAME))["count"]
    logging.info(f"Number of documents in index: {num_docs_in_index}")
    response = query_engine.query(args.query)

    logging.info(
        "Source Nodes: %s",
        json.dumps(
            [node.node.to_dict()["text"] for node in response.source_nodes], indent=4
        ),
    )

    if args.synthesize_response:
        logging.info("Synthesized Response: %s", response.response)

    if args.use_evaluators:
        evaluators: Dict[str, BaseEvaluator] = {
            "faithfulness": FaithfulnessEvaluator(),
            "relevancy": RelevancyEvaluator(),
        }
        eval_results = {
            name: evaluator.evaluate_response(query=args.query, response=response)
            for name, evaluator in evaluators.items()
        }
        eval_results_str = ", ".join(
            "%s: %s" % (name, result.passing) for name, result in eval_results.items()
        )
        logging.info("Evaluation Results: %s", eval_results_str)


def low_level_main():
    """
    Main function for executing a low-level Elasticsearch query.

    This function reads the configuration from a YAML file, constructs an Elasticsearch query
    based on the provided arguments, and retrieves the matching texts from the Elasticsearch index.

    Args:
        None

    Returns:
        None
    """

    logging.basicConfig(level=logging.INFO)
    args = parse_main_args()
    config = AppConfig.from_yaml(args.config_file_path)

    INDEX_NAME = config.ingest.elasticsearch_config.index_name
    INDEX_EMBEDDING_FIELD = config.ingest.elasticsearch_config.index_embedding_field
    EMBEDDING_MODEL_ID = config.ingest.elasticsearch_config.embedding_model_id
    TEXT_FIELD = config.ingest.elasticsearch_config.index_text_field

    client = get_elasticsearch_client_from_env()
    with open("elastic/elastic_templates/query_template.json") as f:
        query_template = json.load(f)
    query_body = replace_placeholders_in_JSON_template(
        query_template,
        embedding_model_id=EMBEDDING_MODEL_ID,
        index_embedding_field=INDEX_EMBEDDING_FIELD,
        query=args.query,
    )
    response = client.search(index=INDEX_NAME, query=query_body)
    retrieved_texts = [hit["_source"][TEXT_FIELD] for hit in response["hits"]["hits"]]
    print(retrieved_texts)


if __name__ == "__main__":
    asyncio.run(main())
    low_level_main()
