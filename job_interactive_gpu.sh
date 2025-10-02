#!/usr/bin/env bash

srun --partition=gpu-shared  --pty --account=gat100 \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 \
    --mem=96G --gpus=1 \
    -t 00:30:00 --wait=0 --export=ALL /bin/bash
