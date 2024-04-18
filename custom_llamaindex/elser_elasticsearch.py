import uuid
from logging import getLogger
from typing import Any, List, Optional

from pydantic import Field
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.elasticsearch import ElasticsearchStore  # type: ignore
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import BulkIndexError, async_bulk


logger = getLogger(__name__)


class ElserElasticsearchStore(ElasticsearchStore):
    """Elasticsearch vector store."""

    pipeline_name: str = Field(
        ...,
        description="Name of the Elasticsearch pipeline to use for inference",
    )
    model_id: str = Field(
        default=".elser_model_1",
        description="ID of the ELSER model to use for inference",
    )
    model_text_field: str = Field(
        default="body_content_field",
        description="Name of the field in the pipeline that contains the text to use for inference",
    )

    def __init__(self, *args, **kwargs):
        pipeline_name = kwargs.pop("pipeline_name")
        model_id = kwargs.pop("model_id", None)
        model_text_field = kwargs.pop("model_text_field", None)
        super().__init__(*args, **kwargs)
        self.pipeline_name = pipeline_name
        self.model_id = model_id or ".elser_model_1"
        self.model_text_field = model_text_field or "body_content_field"

    async def async_add(
        self,
        nodes: List[BaseNode],
        *,
        create_index_if_not_exists: bool = True,
        create_pipeline_if_not_exists: bool = True,
        **add_kwargs: Any,
    ) -> List[str]:
        """Asynchronous method to add nodes to Elasticsearch index.

        Args:
            nodes: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create
                                        the AsyncElasticsearch index if it
                                        doesn't already exist.
                                        Defaults to True.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ImportError: If elasticsearch python package is not installed.
            BulkIndexError: If AsyncElasticsearch async_bulk indexing fails.
        """

        if len(nodes) == 0:
            return []

        if create_index_if_not_exists:
            dims_length = len(nodes[0].get_embedding())
            await self._create_index_if_not_exists(
                index_name=self.index_name, dims_length=dims_length
            )

        if create_pipeline_if_not_exists:
            await self._create_pipeline_if_not_exists()

        requests = []
        return_ids = []
        for node in nodes:
            _id = node.node_id if node.node_id else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": _id,
                "_source": {
                    self.text_field: node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": node_to_metadata_dict(node, remove_text=True),
                },
                "pipeline": self.pipeline_name,
            }
            requests.append(request)
            return_ids.append(_id)
        try:
            success, failed = await async_bulk(
                self.client, requests, chunk_size=self.batch_size, refresh=True
            )
            logger.debug(f"Added {success} and failed to add {failed} texts to index")

            logger.debug(f"added texts {return_ids} to index")
            return return_ids
        except BulkIndexError as e:
            logger.error(f"Error adding texts: {e}")
            firstError = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First error reason: {firstError.get('reason')}")
            raise

    async def _create_pipeline_if_not_exists(self) -> None:
        try:
            await self.client.ingest.get_pipeline(id=self.pipeline_name)
            logger.debug(
                f"Pipeline {self.pipeline_name} already exists. Skipping creation."
            )
        except NotFoundError:
            pipeline_settings = {
                "description": "Inference pipeline using ELSER model",
                "processors": [
                    {
                        "inference": {
                            "field_map": {self.text_field: self.model_text_field},
                            "model_id": self.model_id,
                            "target_field": "ml",
                            "inference_config": {
                                "text_expansion": {"results_field": "tokens"}
                            },
                        }
                    }
                ],
                "version": 1,
            }
            logger.debug(
                f"Creating pipeline {self.pipeline_name} that uses ELSER model"
            )
            await self.client.ingest.put_pipeline(
                id=self.pipeline_name, body=pipeline_settings
            )

    async def _create_index_if_not_exists(
        self, index_name: str, dims_length: int
    ) -> None:
        """Create the AsyncElasticsearch index if it doesn't already exist.

        Args:
            index_name: Name of the AsyncElasticsearch index to create.
            dims_length: Length of the embedding vectors.
        """
        if (
            dims_length > 1
        ):  # MockEmbedding when embeddings are disabled creates embeddings with dim length 1
            return await super()._create_index_if_not_exists(index_name, dims_length)

        if await self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")
        else:
            logger.info(
                "Creating index designed for ELSER embeddings since llama-index embeddings are disabled."
            )
            index_settings = {
                "mappings": {
                    "properties": {
                        "ml.tokens": {"type": "rank_features"},
                        self.text_field: {"type": "text"},
                    }
                }
            }
            logger.debug(
                f"Creating index {index_name} with mappings {index_settings['mappings']}"
            )
            await self.client.indices.create(index=index_name, **index_settings)
