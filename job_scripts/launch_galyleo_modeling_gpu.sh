#!/usr/bin/env bash

galyleo launch --account gat100 --partition gpu-shared \
	--cpus 10 --memory 92 --gpus 1 --time-limit 01:00:00 \
	--interface lab --notebook-dir $HOME/models/fdp-demo \
        --conda-env fdp-demo
