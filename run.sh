#!/bin/bash

printf "Running the dataset generator (Toksearch)...\n"
fdp run python src/dataset_gen.py

printf "Running training script...\n"
python src/train.py

printf "Running testing script...\n"
python src/test.py
