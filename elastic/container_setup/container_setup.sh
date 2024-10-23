# Remove persistent volume from prior setup and set up elasticsearch service
# Be sure to download the security certificate and copy the password into .env before running any python script
docker volume rm es-config
docker run \
--name "elasticsearch" \
--network "elastic" \
--publish "9200:9200" \
--volume "es-config:/usr/share/elasticsearch/config:rw" \
--interactive \
--tty \
--rm \
-it \
-m 8GB \
"docker.elastic.co/elasticsearch/elasticsearch:8.10.4"

# The bottom two services are only required for using the web crawler service on Elasticsearch

# Setup Kibana service
docker run \
--name "kibana" \
--network "elastic" \
--publish "5601:5601" \
--interactive \
--tty \
--rm \
--env "ENTERPRISESEARCH_HOST=http://enterprise-search:3002" \
"docker.elastic.co/kibana/kibana:8.10.4"

# Setup Enterprise Search service
docker run \
--name "enterprise-search" \
--network "elastic" \
--publish "3002:3002" \
--volume "es-config:/usr/share/enterprise-search/es-config:ro" \
--interactive \
--tty \
--rm \
--env "secret_management.encryption_keys=[c34d38b3a14956121ff2170e5030b471551370178f43e5626eec58b04a30fae2]" \
--env "allow_es_settings_modification=true" \
--env "elasticsearch.host=https://172.18.0.2:9200" \
--env "elasticsearch.username=admin" \
--env "elasticsearch.password=*************" \
--env "elasticsearch.ssl.enabled=true" \
--env "elasticsearch.ssl.certificate_authority=/usr/share/enterprise-search/es-config/certs/http_ca.crt" \
--env "kibana.external_url=http://0.0.0.0:5601" \
"docker.elastic.co/enterprise-search/enterprise-search:8.10.4"

