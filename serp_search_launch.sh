#!/usr/bin/env bash

search_url=https://serpapi.com/search
serp_api_key="f82ece71053979e5234ca5f1ecc48bd31c55b2e28c2eae26d0b3439322b8580b" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
