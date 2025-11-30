#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: Please provide n value as argument"
    exit 1
fi

N=$1
TIME_LIMIT=300

if [ $((N % 2)) -ne 0 ]; then
    echo "Error: n must be even"
    exit 1
fi

echo ""
echo "Running all models for n $N"
echo "Time limit: $TIME_LIMIT seconds"
echo ""

# Run CP model
echo ""
echo "----- Starting CP -----"
echo ""
python3 source/CP/run_cp.py -n $N -t $TIME_LIMIT --solver all
echo ""
echo "----- CP model completed -----"
echo ""

# Run SAT model
echo ""
echo "----- Starting SAT  -----"
echo ""
python3 source/SAT/run_sat.py -n $N --save_json --all
echo ""
echo "----- SAT model completed -----"
echo ""

# Run SMT model
echo ""
echo "----- Starting SMT  -----"
echo ""
python3 source/SMT/run_smt.py -n $N --solver all --mode all
echo ""
echo "----- SMT model completed -----"
echo ""

# Run MIP model
echo ""
echo "----- Starting MIP -----"
echo ""
python3 source/MIP/run_mip.py --mode fix_n --n $N --time_limit $TIME_LIMIT
echo ""
echo "----- MIP model completed -----"
echo ""


echo ""
echo "----- All models completed -----"
echo ""
