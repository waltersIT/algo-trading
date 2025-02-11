#!/bin/bash
path="/mnt/c/Users/devea/Desktop/algo-trading"

cd "$path/scripts" || exit

source activate.sh

cd "$path" || exit

python main.py
