"""
This script is used to set up the elasticsearch.Elasticsearch environment.
The script can be used to 
    1) Start the 30 day trial for Elastic License.
    2) Download and deploy the pretrained ELSER model.
    3) Create an index with the ELSER ingestion pipeline.

Author: David Sheng
Date: 2024-03-05
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Tuple, Optional

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import elasticsearch
import elastic_transport

from configs import AppConfig
from elastic.utils import (
    replace_placeholders_in_JSON_template,
    get_elasticsearch_client_from_env,
)

logger = logging.getLogger(__name__)


def start_trial_license(
    client: elasticsearch.Elasticsearch,
) -> elastic_transport.ObjectApiResponse:
    """
    Starts the 30-day trial for the Elastic License.

    Args:
        client (elasticsearch.Elasticsearch): The Elasticsearch client.

    Returns:
        elastic_transport.ObjectApiResponse: The response from starting the trial license.

    Raises:
        Exception: If an error occurs when activating the trial license.
    """
    logger.info("Starting the 30 day trial for Elastic License...")
    try:
        response = client.license.post_start_trial(acknowledge=True)
        logger.info(response)
    except Exception as e:
        logger.warning(f"An error occurred when activating the trial license: {e}")
        response = e
        pass
    return response


def download_model(
    client: elasticsearch.Elasticsearch, model_id: str, model_text_field: str
) -> elastic_transport.ObjectApiResponse:
    """
    Downloads a trained model from elasticsearch.Elasticsearch if it doesn't already exist.

    Args:
        client (elasticsearch.Elasticsearch): The elasticsearch.Elasticsearch client.
        model_id (str): The ID of the trained model.
        model_text_field (str): The name of the text field used by the model.

    Returns:
        elastic_transport.ObjectApiResponse: The response from elasticsearch.Elasticsearch.

    """
    try:
        existing_trained_models = client.ml.get_trained_models(
            model_id=model_id, allow_no_match=True
        )
        logger.info("Model already downloaded...")
        return existing_trained_models.body
    except:
        model_schema = {"input": {"field_names": [model_text_field]}}
        logger.info("Downloading the ELSER model...")
        response = client.ml.put_trained_model(model_id=model_id, body=model_schema)
        time.sleep(90)  # Wait for the model to be downloaded
        return response


def deploy_model(
    client: elasticsearch.Elasticsearch,
    model_id: str,
    deployment_id: Optional[str] = None,
) -> str:
    """
    Deploys the ELSER model.

    Args:
        client (elasticsearch.Elasticsearch): The elasticsearch.Elasticsearch client.
        model_id (str): The ID of the model to deploy.
        deployment_id (str, optional): The ID to use for the deployment. Defaults to the model_id.

    Returns:
        str: The deployment ID of the deployed model.
    """
    if not deployment_id:
        deployment_id = model_id

    logger.info("Deploying the ELSER model...")
    existing_deployments = (
        client.ml.get_trained_models_stats(model_id=model_id)
        .body["trained_model_stats"][0]
        .get("deployment_stats")
    )
    if (
        existing_deployments
        and existing_deployments.get("deployment_id") == deployment_id
    ):
        logger.info("Model already deployed...")
        return deployment_id
    else:
        client.ml.start_trained_model_deployment(
            model_id=model_id, deployment_id=deployment_id
        )
        return deployment_id



@dataclasses.dataclass
class SetupScriptArgs:
    """
    Represents the arguments for the setup script.

    Args:
        config_file_path (str): The path to the configuration file.
        start_trial_license (bool): Whether to start the trial license.
        download_and_deploy_pretrained_model (bool): Whether to download and deploy a pretrained model.
        create_elser_ingestion_index (bool): Whether to create the Elser ingestion index.
    """

    config_file_path: str
    start_trial_license: bool
    download_and_deploy_pretrained_model: bool
    #create_elser_ingestion_index: bool


def parse_main_args() -> SetupScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        "-c",
        default="configs/ibm_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--start_trial_license",
        "-s",
        action="store_true",
        help="Whether to start the trial license",
    )
    parser.add_argument(
        "--download_and_deploy_pretrained_model",
        "-d",
        action="store_true",
        help="Whether to download and deploy the pretrained model",
    )

    args = parser.parse_args()
    return SetupScriptArgs(
        config_file_path=args.config_file_path,
        start_trial_license=args.start_trial_license,
        download_and_deploy_pretrained_model=args.download_and_deploy_pretrained_model,
        # create_elser_ingestion_index=args.create_index,
    )


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_main_args()
    client = get_elasticsearch_client_from_env(timeout=180)

    config_file_path = args.config_file_path
    config = AppConfig.from_yaml(config_file_path).ingest.elasticsearch_config
    INDEX_NAME = config.index_name
    EMBEDDING_MODEL_ID = config.embedding_model_id
    EMBEDDING_MODEL_TEXT_FIELD = config.embedding_model_text_field

    if args.start_trial_license:
        start_trial_license(client)
    if args.download_and_deploy_pretrained_model:
        download_model(
            client,
            model_id=EMBEDDING_MODEL_ID,
            model_text_field=EMBEDDING_MODEL_TEXT_FIELD,
        )
        deploy_model(client, model_id=EMBEDDING_MODEL_ID)

if __name__ == "__main__":
    main()

