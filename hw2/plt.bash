#!/bin/bash

declare -a arr
for file in data/hc*
do
    arr=("${arr[@]}" "$file")
done

python plot.py "${arr[@]}"

