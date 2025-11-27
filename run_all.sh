#!/bin/bash

TIME_LIMIT=300

# Run CP model for all n
echo ""
echo "----- Starting CP -----"
echo ""
python3 source/CP/run_cp.py -t $TIME_LIMIT --solver all
echo ""
echo "----- CP completed -----"
echo ""

# Run SAT model for all n
echo ""
echo "----- Starting SAT -----"
echo ""
python3 source/SAT/run_sat.py --save_json --all
echo ""
echo "----- SAT model completed -----"
echo ""

# Run SMT model for all n
echo ""
echo "----- Starting SMT -----"
echo ""
python3 source/SMT/smt_model.py -all --solver all --mode all
echo ""
echo "----- SMT model completed -----"
echo ""

# Run MIP model for all n
echo ""
echo "----- Starting MIP -----"
echo ""
python3 source/MIP/run_mip.py --mode all --time_limit $TIME_LIMIT
echo ""
echo "----- MIP model completed -----"
echo ""

echo ""
echo "All models completed for all n values"
echo ""
