podman run \
--name "enterprise-search" \
--publish "3002:3002" \
--network elastic \
--volume "/Users/dsheng/Documents/GitHub/watsonx-wxd-setup-and-ingestion/docker_setup/enterprise_search/config:/usr/share/enterprise-search/es-config:ro" \
--interactive \
--tty \
--rm \
"docker.elastic.co/enterprise-search/enterprise-search:8.10.4"