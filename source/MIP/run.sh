#!/bin/bash

n=$1
solver_type="highs"
opt="feasible" 
sym="fixWeek2" #"circleMethod" #"fixWeek2"
time_limit=300
solver_verbose=1
json_path='./res/MIP'

python3 source/MIP/run_mip.py --n $n --time_limit $time_limit --solver_type $solver_type --opt $opt --sym $sym --solver_verbose $solver_verbose --json_path $json_path
