#!/bin/bash

# ============================================================
#  CHECK.SH — versione funzionante per solver.py attuale
#  Genera tutte le combinazioni per un singolo n pari passato
# ============================================================

if [ $# -ne 1 ]; then
    echo "Usage: $0 <even_n>"
    exit 1
fi

n=$1

if [ $((n % 2)) -ne 0 ]; then
    echo "Error: n must be even"
    exit 1
fi

TIMEOUT=300
SOLVER="C:/Users/asper/Desktop/STS-CDMO/source/SAT/solver.py"
VERBOSE=true
SAVE_JSON=true

EX1_LIST=("np" "heule")
AMK_LIST=("np" "seq" "totalizer")
SB_LIST=("sb" "no_sb")
MODE_LIST=("decisional" "optimization")

# ============================================================
# Funzione: esegue una combinazione
# ============================================================
run_combo() {
    local n=$1
    local mode=$2
    local ex1=$3
    local amk=$4
    local sb=$5

    echo ""
    echo "----------------------------------------------------"
    echo " Running: n=$n | mode=$mode | ex1=$ex1 | amk=$amk | sb=$sb"
    echo "----------------------------------------------------"

    CMD="python \"$SOLVER\" -n $n --exactly_one_encoding $ex1 --at_most_k_encoding $amk"

    if [ "$sb" == "sb" ]; then
        CMD="$CMD --sb"
    else
        CMD="$CMD --no_sb"
    fi

    if [ "$mode" == "optimization" ]; then
        CMD="$CMD --run_optimization"
    else
        CMD="$CMD --run_decisional"
    fi

    CMD="$CMD --timeout $TIMEOUT"

    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi

    if [ "$SAVE_JSON" = true ]; then
        CMD="$CMD --save_json"
    fi

    eval $CMD
}

# ============================================================
# Loop principale sulle combinazioni
# ============================================================
for mode in "${MODE_LIST[@]}"; do
    for ex1 in "${EX1_LIST[@]}"; do
        for amk in "${AMK_LIST[@]}"; do
            for sb in "${SB_LIST[@]}"; do
                run_combo $n $mode $ex1 $amk $sb
            done
        done
    done
done

echo ""
echo "ALL TESTS COMPLETED. Results saved in ./res/SAT/$n.json"
