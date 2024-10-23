import json
import logging
import os
import re
from pathlib import Path

import dotenv
import elasticsearch

logger = logging.getLogger(__name__)

def _get_elasticsearch_credentials_from_env(
    dotenv_path: str | Path = "elastic/.env",
) -> tuple[str, str, str, str]:
    """
    Retrieves the Elasticsearch credentials from environment variables.

    This function retrieves the necessary configuration values from environment variables
    and returns them as a tuple.

    Returns:
        tuple[str, str, str, str]: A tuple containing the Elasticsearch credentials.

    Raises:
        EnvironmentError: If any of the required environment variables are not set.
    """
    dotenv.load_dotenv(dotenv_path)
    url = os.environ.get("ELASTIC_URL")
    username = os.environ.get("ELASTIC_USERNAME")
    password = os.environ.get("ELASTIC_PASSWORD")
    cert_path = os.environ.get("ELASTIC_CERT_PATH")
    if any([url is None, username is None, password is None, cert_path is None]):
        raise EnvironmentError(
            "The following environment variables must be set: ELASTIC_URL, ELASTIC_USERNAME, ELASTIC_PASSWORD, ELASTIC_CERT_PATH"
        )
    return url, username, password, cert_path

def get_elasticsearch_client_from_env(**kwargs) -> elasticsearch.Elasticsearch:
    """
    Get an Elasticsearch client using environment variables.

    This function retrieves the necessary configuration values from environment variables
    and creates an Elasticsearch client with the provided configuration.

    Args:
        **kwargs: Additional keyword arguments to be passed to the Elasticsearch client.

    Returns:
        Elasticsearch: An instance of the Elasticsearch client.
    """
    url, username, password, cert_path = _get_elasticsearch_credentials_from_env()
    client = elasticsearch.Elasticsearch(
        url, ca_certs=cert_path, basic_auth=(username, password), **kwargs
    )
    client.info()  # Check if client is working
    return client

def get_async_elasticsearch_client_from_env(
    **kwargs,
) -> elasticsearch.AsyncElasticsearch:
    """
    Retrieves an asynchronous Elasticsearch client using environment variables.

    This function retrieves the necessary configuration values from environment variables
    and creates an Elasticsearch client with the provided configuration.

    Args:
        **kwargs: Additional keyword arguments to be passed to the AsyncElasticsearch constructor.

    Returns:
        AsyncElasticsearch: An asynchronous Elasticsearch client.
    """
    url, username, password, cert_path = _get_elasticsearch_credentials_from_env()
    get_elasticsearch_client_from_env()  # Check if client is working
    return elasticsearch.AsyncElasticsearch(
        url, ca_certs=cert_path, basic_auth=(username, password), **kwargs
    )

def create_api_key(
    client: elasticsearch.Elasticsearch, name: str = "my_api_key"
) -> dict:
    """
    Creates and API key for Elasticsearch with all priveleges.

    Args:
        client (Elasticsearch): An Elasticsearch client instance.
        username (str): The username for which to retrieve the API key.

    Returns:
        dict: A dictionary containing the API key for the specified user.
    """
    body = {
        "name": name,
        "role_descriptors": {
            "role_name": {
                "cluster": ["all"],
                "index": [{"names": ["*"], "privileges": ["all"]}],
            }
        },
    }
    response = client.security.create_api_key(body=body)
    return response

def replace_placeholders_in_JSON_template(JSON_template: dict, **kwargs: str) -> dict:
    """
    Replaces placeholders in a JSON template with provided values.

    Args:
        JSON_template (dict): The JSON template with placeholders.
        **kwargs (str): Keyword arguments representing the values to replace the placeholders with.

    Returns:
        dict: The modified JSON with placeholders replaced by the provided values.
    """
    template_str = json.dumps(JSON_template)
    for key, value in kwargs.items():
        placeholder = "{{" + key + "}}"
        template_str = template_str.replace(placeholder, value)

    remaining_placeholders = re.findall(r"\{\{.*?\}\}", template_str)
    if remaining_placeholders:
        logger.warning(f"Remaining placeholders found: {remaining_placeholders}")

    return json.loads(template_str)

if __name__ == "__main__":
    client = get_elasticsearch_client_from_env()
    print(client.info())
