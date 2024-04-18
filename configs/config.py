from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import Dict

import yaml


class FileStoreType(Enum):
    COS = "cos"
    LOCAL = "local"


@dataclass
class FileStoreConfig:
    type: FileStoreType
    location: str
    service_credentials_path: str
    num_files_to_ingest: int


@dataclass
class ElasticsearchConfig:
    index_name: str
    index_text_field: str
    index_embedding_field: str
    pipeline_name: str
    embedding_model_id: str
    embedding_model_text_field: str


@dataclass
class IngestConfig:
    file_store_config: FileStoreConfig
    elasticsearch_config: ElasticsearchConfig
    chunk_size: int
    chunk_overlap: int


@dataclass
class QueryConfig:
    num_docs_to_retrieve: int
    llm_path: str
    prompt_template_path: str


@dataclass
class AppConfig:
    ingest: IngestConfig
    query: QueryConfig

    @classmethod
    def from_yaml(cls, yaml_file_path: str):
        with open(yaml_file_path, "r") as f:
            data = yaml.safe_load(f)

        file_store_config = FileStoreConfig(**data["ingest"]["file_store"])
        elasticsearch_config = ElasticsearchConfig(**data["ingest"]["elasticsearch"])
        ingest_config = IngestConfig(
            file_store_config=file_store_config,
            elasticsearch_config=elasticsearch_config,
            chunk_size=data["ingest"]["chunk_size"],
            chunk_overlap=data["ingest"]["chunk_overlap"],
        )
        query_config = QueryConfig(**data["query"])

        return cls(ingest=ingest_config, query=query_config)


class LLMService(Enum):
    BAM = "BAM"
    WML = "WML"


@dataclass
class LLMConfig:
    service_name: LLMService
    llm_id: str
    llm_params: Dict

    @classmethod
    def from_json(cls, file_path: str) -> "LLMConfig":
        with open(file_path, "r") as file:
            data = json.load(file)
        if "service_name" not in data:
            raise ValueError("service_name must be specified in the JSON file.")
        if "model_name" not in data:
            raise ValueError("model_name must be specified in the JSON file.")
        return cls(
            service_name=LLMService(data["service_name"]),
            llm_id=data["model_name"],
            llm_params=data.get("model_params"),
        )

    @classmethod
    def from_json_path(cls, file_path: str) -> "LLMConfig":
        _, ext = os.path.splitext(file_path)
        if ext.lower() != ".json":
            raise ValueError("If a path is provided, it must be a JSON file.")
        return cls.from_json(file_path)
