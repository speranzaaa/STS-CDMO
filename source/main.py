import solver
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_teams", type=str, default="6-20")
    parser.add_argument("-t", "--timeout", type=int, default=300)
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--run_decisional", action="store_true")
    parser.add_argument("--run_optimization", action="store_true")
    parser.add_argument("--sb", action="store_true")
    args = parser.parse_args()

    # Converti input in lista di numeri pari
    n_teams = solver.parse_n_teams([args.n_teams])

    # encoding predefiniti
    eo_func = solver.eo_exactly_one_seq
    ak_func = solver.amk_seq

    for n in n_teams:
        if args.run_decisional:
            res_dec = solver.solve_sts_decisional(
                n=n,
                max_diff_k=n-1,
                timeout_seconds=args.timeout,
                exactly_one_encoding=eo_func,
                at_most_k_encoding=ak_func,
                symmetry_breaking=args.sb,
                verbose=True,
            )
            if args.save_json:
                solver.save_results_as_json(n, res_dec, model_name=f"decisional_seq_seq", output_dir="res/SAT")

        if args.run_optimization:
            res_opt = solver.solve_sts_optimization(
                n=n,
                timeout_seconds=args.timeout,
                exactly_one_encoding=eo_func,
                at_most_k_encoding=ak_func,
                symmetry_breaking=args.sb,
                verbose=True,
            )
            if args.save_json:
                solver.save_results_as_json(n, res_opt, model_name=f"optimization_seq_seq", output_dir="res/SAT")

if __name__ == "__main__":
    main()
