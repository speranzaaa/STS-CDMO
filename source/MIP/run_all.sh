#!/bin/bash

n=$1
solver_types=("cbc" "highs" "sciPy")
opts=("homeAwayDiff0" "homeAwayDiff1" "feasible")
syms=("circleMethod" "fixWeek1" "fixWeek2" "noSym")
time_limit=300
solver_verbose=0
json_path="./res/MIP"
for solver_type in "${solver_types[@]}"; do
    for opt in "${opts[@]}"; do
        for sym in "${syms[@]}"; do
            echo "solver: $solver_type , opt: $opt , sym: $sym"
            python3 source/MIP/run_mip.py \
            --n $n \
            --time_limit $time_limit \
            --solver_type $solver_type \
            --opt $opt \
            --sym $sym \
            --json_path $json_path 
        done
    done
done