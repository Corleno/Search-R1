#!/usr/bin/env bash

search_url=https://serpapi.com/search
serp_api_key="252eaa31091c32b34d36184f2a1c17a9647bce6cfebdab837c13137dad17f1bd" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
