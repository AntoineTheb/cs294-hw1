#!/bin/bash

declare -a arr
for file in data/hc_best*
do
    arr=("${arr[@]}" "$file")
done

python plot.py "${arr[@]}"

