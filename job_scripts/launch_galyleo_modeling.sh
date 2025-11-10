#!/usr/bin/env bash

galyleo launch --account gat100 --partition shared \
	--cpus 8 --memory 16 --time-limit 02:00:00 \
	--interface lab --notebook-dir $HOME/models/fdp-demo \
        --conda-env fdp-demo
