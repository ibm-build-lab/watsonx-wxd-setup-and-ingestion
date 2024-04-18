# Setting Up and Ingesting Files into Elasticsearch

This repository contains Python 3.10 code assets designed to configure an instance of Elasticsearch and ingest files into an index in Elasticsearch. The stored files are embedded using Elastic's ELSER model and can be used for use-cases such as Retrieval Augmented Generation.

Currently supported file types:
| File Type | Loader |
| --- | --- |
| `.pdf` | [LlamaHub PDF Loader](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers) |
| `.docx` | [LlamaHub Docx Loader](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers) |
| `.pptx` | [LlamaHub Pptx Loader](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers) |
| `.txt` | [LlamaHub Unstructured File Loader](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers) |
| `.html` | [LlamaHub Unstructured File Loader](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers) |

Currently supported file locations:
1. Local directories
1. Cloud Object Storage

## Table of Contents

- [Setting Up and Ingesting Files into Elasticsearch](#setting-up-and-ingesting-files-into-elasticsearch)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Using The Repository](#using-the-repository)
    - [Setting up the environment](#setting-up-the-environment)
    - [Connecting to Elasticsearch](#connecting-to-elasticsearch)
    - [Sourcing your documents](#sourcing-your-documents)
      - [Customize the config YAML file](#customize-the-config-yaml-file)
      - [Setup and Ingest into Elastisearch](#setup-and-ingest-into-elastisearch)
    - [Querying Your Data](#querying-your-data)
      - [High-Level Querying](#high-level-querying)
      - [Low-Level Querying](#low-level-querying)
  - [Sample Data](#sample-data)
    - [Nvidia Q\&A Text Files](#nvidia-qa-text-files)
    - [IBM Watsonx.ai Sales Documents](#ibm-watsonxai-sales-documents)

## Prerequisites

This repository assumes that you have an instance of Databases for Elasticsearch Platinum edition deployed on IBM Cloud. The Platinum edition is required to use the ELSER model in Elasticsearch, which is leveraged for semantic search. If this is not available, you can also test this repository using Elasticsearch deployed on an Openshift cluster or locally. For guidance on deploying Elasticsearch locally, refer to the [official Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html), or use the Dockerfile provided in the `container_setup` directory.

## Using The Repository

### Setting up the environment

To start using the repository, first clone the repository into a local directory. Then, create a virtual environment and install the requirements to run the Python scripts onto the environment. Run the code provided below to create the virtual environment and install the ncessary requirements

``` shell
python3 -m venv .venv
```
```shell
source .venv/bin/activate
```
```shell
pip install -r requirements.txt
```

When running code from the repository, make sure that the environment you have created is active

### Connecting to Elasticsearch

Once you have set up the repository with the necessary requirements, the next step is to authenticate to your Elasticsearch service.

To authenticate to Elasticsearch, you will need:

- username
- password
- hostname
- port
- SSL certificate

<details> 
<summary> Expand for details on how to get these values from Databases for Elasticsearch on IBM Cloud </summary>

If you have a Databases for Elasticsearch configured on IBM Cloud, you can get these values by going to your resource list in IBM Cloud and selecting Elasticsearch under your database resources. 

1. In the overview tab of your Elasticsearch resource, scroll to the bottom and select the https endpoint. Here, you will find your hostname, port, and SSL certificate.

    <img src="images/get_cert_and_route.png" alt="Location of hostname and certificate" width="1800"/>

    Note the hostname and the port, and save a copy of the SSL certificate to a directory in this repostiory

1. Next, go top the service credentials tab and expand the service credentials. Save the username and password.

    <img src="images/get_user_and_pass.png" alt="Location of username and password" width="1800"/>

    </details>


Once you have these credentials, make a .env file containing these credentials. To do so, go to the ```elastic``` folder inside the repository and copy the contents of the ```.envExample``` file into a new file called ```.env```. To populate this file:

- Replace the value after ```ELASTIC_URL``` with ```https://<hostname>:<port>``` 
- Replace the value after ```ELASTIC_USERNAME``` and ```ELASTIC_PASSWORD``` with the username and password for Elasticsearch
- Copy the relative path of the SSL certificate you copied into the repository, and replace the value after ```ELASTIC_CERT_PATH``` with the relative path

(Optional) To verify whether you are able to connect to Elasticsearch, try to instantiate an Elasticsearch Python client and test its connection. The command below should run without error if the values in the .env file are inputted correctly

``` shell
python3 elastic/utils.py
```

### Sourcing your documents
The next step is to source your documents that you wish to ingest into Elasticsearch. documents can be ingested into Elasticsearch via a directory on your local machine or through a bucket in Cloud Object Storage on IBM Cloud. For details on setting up a Cloud Object Storage bucket, refer to this documentation [README in the COS folder](COS/README.md). Currently, the supported file types are `.pdf, .txt, .docx, .pptx`. Note that not all `.pptx` files may be supported. If you wish to ingest documents through a local directory, save all the documents to a directory and note the path of the directory. If you wish to ingest documents through Cloud Object Storage, load the files documents into a bucket in your configured instance of Cloud Object Storage and save the name of the bucket. A small collection of sample documents is provided below in the [sample data section](#sample-data).

#### Customize the config YAML file

The scripts for setting up Elasticsearch and ingesting your documents can be configured via a YAML config file. A sample config is provided in the `configs` folder of the repository. Using the `sample_config.yaml` file, create a new config file and populate the values as described below


| Field                                  | Default Value                                         | Description                                                                                       |
|----------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `ingest.file_store.type`               | `local`                                               | The type of file store. Can be `cos` (Cloud Object Storage) or `local`.                           |
| `ingest.file_store.location`           | `data/nvidia/answers`                                 | If `cos`, the name of the bucket. If `local`, the directory path.                                 |
| `ingest.file_store.service_credentials_path` | `null`                                      | The path to the service credentials if `cos` is the type of file store.                                                              |
| `ingest.file_store.num_files_to_ingest` | `500`                                                | The number of files to ingest.                                                                    |
| `ingest.elasticsearch.index_name`      | `index-created-in-setup-ingestion-repo-sample-config` | The name of the index to ingest into.                                                             |
| `ingest.elasticsearch.index_text_field`| `body_content_field`                                  | The name of the field in the index used to store document text.                                   |
| `ingest.elasticsearch.index_embedding_field` | `sparse_embedding` | The name of the field in the index used to store embeddings. 
| `ingest.elasticsearch.pipeline_name`   | `elser_ingestion`                                     | The name of the pipeline to use for ingestion.                                                    |
| `ingest.elasticsearch.embedding_model_id` | `.elser_model_1`                                 | The name of the embedding model to use for ingestion.                                             |
| `ingest.elasticsearch.embedding_model_text_field` | `text_field`                             | The name of the field the embedding model looks for text in.                                      |
| `ingest.chunk_size`                    | `256`                                                 | The number of tokens per chunk.                                                                   |
| `ingest.chunk_overlap`                 | `128`                                                 | The number of tokens to overlap between chunks.                                                   |
| `query.num_docs_to_retrieve`           | `5`                                                   | The number of documents to retrieve on querying.                                                  |
| `query.llm_path`                       | `configs/llm_config/llms/wml_granite_13b_chat_config.json` | The path to the LLM configuration.                                                               |
| `query.prompt_template_path`           | `configs/llm_config/prompt_templates/basic_rag_template.txt` | The path to the prompt template. |
#### Setup and Ingest into Elastisearch
Once you have finished making your config file, copy the path to your config and run the following python script if you need to setup Elastisearch. Skip this step if you already have an Elasticsearch database configured:

 ```shell
 python3 elastic/setup.py [-s -d ] -c "path/to/your/config/file.yaml"
 ```

This script will use the configuration file and does the following in sequence:

1. Try to activate a trial Elasticsearch license if -s is specified, ignores if not
2. Download and deploy the ELSER model from Elastic's servers if -d is specified, ignores if not

To ingest your documentation, run the following:
```python
 python3 elastic/ingest.py -c "path/to/your/config/file.yaml"
 ```
 This script will use the configuration file and does the following in sequence:

1. Creates an index with the index name specified in the config file based on ```elastic/configs/index_config.json``` and a default pipeline with a name specified in the config file based on ```elastic/configs/inference_pipeline_config.json```
2. Use the pipeline to ingest your documents into the index based on the fields under the `ingest` section of the config. 

Once ingestion is complete, your documents are ready to be used for RAG use cases. 

### Querying Your Data

The ```elastic/query.py``` script provides two main scripts to query your Elasticsearch index: 
1. high-level querying using the LlamaIndex framework
2. low-level querying using the Elasticsearch client

#### High-Level Querying
The main() function provides a high-level interface for querying the Elasticsearch index through LlamaIndex. It uses a configuration file to set up the query engine, and optionally synthesizes a response using a language model and evaluates the response using specified evaluators.

To use this functionality, run the script with the following command:

```shell
python elastic/query.py -c "path/to/your/config/file.yaml" -q "your query" -s -e
```

Optional arguments include:

--synthesize_response or -s: If included, the script will synthesize a response using a language model in the config file
--use_evaluators or -e: If included, the script will use the LlamaIndex faithfulness and relevancy evaluators to evaluate the response.

#### Low-Level Querying
The low_level_main() function provides a low-level interface for querying the Elasticsearch index. It directly constructs an Elasticsearch query based on the provided arguments and retrieves the matching texts from the Elasticsearch index.

## Sample Data

### Nvidia Q&A Text Files
To test this repository for basic functionality, a CSV file in ```data/nvidia``` called ```data/nvidia/NvidiaDocumentationQandApairs.csv``` has been provided. This CSV file contains a set of questions and answers related to NVIDIA. If you would like to test the repository with this file, run the following:
1. Generate a collection of documents by converting each answer into an individual document by running ```data/nvidia/nvidia_processing.py```. This script will create a new subfolder called "answers" and write each row in the "answer" column in the CSV to a text file. There are a total of ~7000 rows in this csv. The source for this CSV can be found [here](https://www.kaggle.com/datasets/gondimalladeepesh/nvidia-documentation-question-and-answer-pairs/data)

    ```shell
    python3 data/nvidia/nvidia_processing.py
    ```

2. The ```sample_config.yaml``` file is configured to use the files generated by the script in the previous step. Use the command below to set up an index named ```index-created-in-setup-ingestion-repo-sample-config``` and ingest the generated .txt files into the index.

    ```shell
    python3 elastic/ingest.py -c "elastic/configs/setup_and_ingest_configs/sample_config.yaml"
    ```

3. Test the ingested documents by running 

   ```shell
   elastic/query.py -q "your query in quotes" -c "elastic/configs/setup_and_ingest_configs/sample_config.yaml"
   ```


### IBM Watsonx.ai Sales Documents
To test the repository against a more robust set of documents, here is a link to a [box folder](https://ibm.box.com/s/5lxisye5k379lwn0puisbp1d5g9461h9) that contains a few sales-oriented documents regarding Watsonx.ai from Seismic. This set of documents contains PDF, Docx, and Pptx. With the exception of "watsonx.ai Client Presentation.pptx", the repository is able to handle all the documents in the folder. To use this document set, follow the steps below:
1. Follow the link to the box folder and copy the folder contents to a local directory
1. In the ```ibm_config.yaml``` file in ```configs```, change ```ingest.file_store.location``` value to the path of the folder where the contents were copied to
1. (Optional) Rename the ```ingest.elasticsearch.index_name``` value to another name of choice
2. Use the same scripts ```elastic/setup.py``` to set up and ```elastic/ingest.py``` to ingest documents to Elasticsearch, but this time point the script to the     
    ```shell
    python3 elastic/ingest.py -c "elastic/configs/setup_and_ingest_configs/ibm_config.yaml"
    ```
4. Test the ingested documents by running ```elastic/query.py```. Since a set of questions has not been curated for this document set yet, you can ask a specific query using the ```-q <YOUR_QUERY>``` argument while running the Python script. 
   ```shell
   python3 elastic/query.py -c "elastic/configs/setup_and_ingest_configs/ibm_config.yaml" -q "your query in quotes"
   ```
5. (Optional) You can also test conversational search using the [BAM api](https://bam.res.ibm.com/) using the retrieved documents. To do so, create a ```.env``` file in the root folder following the ```.envExample``` file and put your personal BAM API key in the BAM_APIKEY field. To test conversational search after querying, append the ```-e``` argument when running ```query.py```.
    ```shell
    python3 elastic/eval/query.py -c "elastic/configs/setup_and_ingest_configs/ibm_config.yaml" -q "your query in quotes" -e 
    ```
