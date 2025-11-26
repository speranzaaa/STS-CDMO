#!/bin/bash
SOLVER="./generator.py"

run_smt_all() {

  echo "Running all instances"

  CMD="python3 \"$SOLVER\" -all"

  eval $CMD
}

run_smt_all