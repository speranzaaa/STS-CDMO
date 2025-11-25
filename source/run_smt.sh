#!/bin/bash
SOLVER=".source/SMT/generator.py"


run_smt_all() {

  echo "Running all instances"

  CMD="python \"$SOLVER\" -all"

  eval $CMD
}

run_smt_all