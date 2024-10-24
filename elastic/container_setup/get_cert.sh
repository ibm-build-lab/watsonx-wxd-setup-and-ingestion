#!/bin/bash
# Get security certificate from Elasticsearch
rm -r certs
docker cp elastic-es01-1:/usr/share/elasticsearch/config/certs .