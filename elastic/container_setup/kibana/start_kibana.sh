podman network create elastic
podman run --name kibana \
    --network elastic \
    -v /Users/dsheng/Documents/GitHub/watsonx-wxd-setup-and-ingestion/container_setup/kibana/config:/usr/share/kibana/config \
    -p 5601:5601 \
    --rm \
    docker.elastic.co/kibana/kibana:8.10.4