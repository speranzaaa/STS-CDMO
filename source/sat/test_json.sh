#!/bin/bash

# Array di squadre (anche numeri pari da 6 a 20)
TEAMS=(6 8 10 12 14 16 18 20)

# Timeout per ogni istanza
TIMEOUT=300

# Directory dove salvare i risultati
OUTPUT_DIR="res/SAT"

# Loop su ciascun numero di squadre
for N in "${TEAMS[@]}"; do
    echo "----------------------------------------"
    echo " Test su n = $N squadre"
    echo "----------------------------------------"

    python solver.py \
        -n $N \
        --all \
        --run_decisional \
        --run_optimization \
        --save_json \
        --timeout $TIMEOUT \
        --sb \
        --verbose

    echo
done

echo "Tutti i test completati!"
echo "Risultati salvati in: $OUTPUT_DIR"