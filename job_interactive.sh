#!/usr/bin/env bash

srun --partition=shared  --pty --account=gat100 \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 \
    --mem=4G -t 01:30:00 --wait=0 --export=ALL /bin/bash
