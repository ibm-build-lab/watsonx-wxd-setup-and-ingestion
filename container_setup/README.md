To deploy enterprise search, Kibana and Elasticsearch in the same docker network, locally on your machine:

1. Copy .envExample to .env and fill in values for 
```
ELASTIC_PASSWORD="changeme"
KIBANA_PASSWORD="changeme"
```

2. Run 
```
docker compose up
```
to deploy ElasticSearch, Enterprise Search and Kibana using Docker.

Alternatively, the `kibana/start_kibana.sh` and `enterprise_search/start_enterprise_search.sh` shell scripts deploy `kibana` and `es` containers via podman instead.
