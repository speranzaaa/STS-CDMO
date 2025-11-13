#!/bin/bash
# ================================================================
# Test automatico per valutare la scalabilità del solver SAT STS
# ================================================================

n_values=(6 8 10 12 14 16 18 20)
output_file="risultati_scalabilita.csv"
exactly_one="seq"
at_most_k="totalizer"
timeout=300


# Intestazione del file CSV
echo "n_teams,time(s),status" > $output_file

# Loop su tutti i valori di n
for n in "${n_values[@]}"; do
    echo "----------------------------------------"
    echo " Test su n = $n squadre"
    echo "----------------------------------------"

    # Misura il tempo di esecuzione in secondi
    START=$(date +%s)

    # Esegui il solver
    python solver.py --n_teams $n --exactly_one_encoding $exactly_one \
    --at_most_k_encoding $at_most_k --sb --run_decisional --timeout $timeout

    # Salva il codice di uscita (0 = successo, diverso = fallimento/timeout)
    STATUS=$?

    END=$(date +%s)
    RUNTIME=$((END - START))

    # Interpreta lo stato
    if [ $STATUS -eq 0 ]; then
        RESULT="solved"
    else
        RESULT="timeout_or_error"
    fi

    # Scrivi il risultato nel file CSV
    echo "$n,$RUNTIME,$RESULT" >> $output_file

    echo "Tempo impiegato: $RUNTIME s ($RESULT)"
    echo
done

echo "Test completati!"
echo "Risultati salvati in: $output_file"