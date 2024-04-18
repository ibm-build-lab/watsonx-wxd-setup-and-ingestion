# %%
import json
import copy

from elasticsearch.helpers import scan, bulk
from langchain.text_splitter import RecurisveCharacterTextSplitter

from elastic.utils import get_client
from elastic.setup import create_index

client = get_client()

# %%
# Define the index and pipeline names
source_index = "search-web-crawl-test"
destination_index = "web-crawl-ml"
pipeline_name = "elser_inference"
with open("elastic/elastic_configs/index_config.json") as f:
    index_settings = json.load(f)
    create_index(client, destination_index, index_settings)


# Get all documents from the source index
documents = scan(client, index=source_index)

# Prepare the documents for the bulk operation
chunk_splitter = RecurisveCharacterTextSplitter(
    separator=". ", chunk_size=1000, chunk_overlap=100, length_function=len
)
actions = []
idx = 0
for doc in documents:
    text = doc["_source"]["body_content"]
    text_chunks = chunk_splitter.split_text(text)
    for chunk in text_chunks:
        idx += 1
        new_doc = copy.deepcopy(doc)  # Copy the original document
        new_doc["_source"][
            "body_content"
        ] = chunk  # Replace the body content with the chunk
        new_doc["_source"]["text"] = chunk
        action = {
            "_index": destination_index,
            "_id": idx,
            "_source": new_doc["_source"],
            "pipeline": pipeline_name,
        }
        actions.append(action)
        # Perform the bulk operation in chunks
        if len(actions) >= 20:  # Adjust the chunk size as needed
            success, failed = bulk(client, actions, raise_on_error=False)
            print(f"Successful operations: {success}")
            print(f"Failed operations: {len(failed)}")
            actions = []
# Perform the bulk operation for the last chunk
if actions:
    success, failed = bulk(client, actions, raise_on_error=False)
    print(f"Successful operations: {success}")
    print(f"Failed operations: {len(failed)}")
