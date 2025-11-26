#!/bin/bash
SOLVER="source/SMT/generator.py"


run_smt_all() {

  echo "Running all instances"

  CMD="python3 \"$SOLVER\" -all -n 6"

  eval $CMD
}

run_smt_all