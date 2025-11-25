from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from itertools import combinations, product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from z3 import *  # noqa: F401 (preserve original behaviour)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def to_binary(num: int, length: Optional[int] = None) -> str:
    """Return binary representation of ``num`` as string. Optionally zero-pad to ``length``."""
    b = bin(num).split("b")[-1]
    if length:
        return b.rjust(length, "0")
    return b


# ---------------------------------------------------------------------------
# Exactly-one encodings
# ---------------------------------------------------------------------------

def at_least_one_np(bool_vars: Sequence[BoolRef]) -> BoolRef:
    """Naive (pairwise) at-least-one: disjunction of the variables."""
    return Or(list(bool_vars))


def at_most_one_np(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    """Naive (pairwise) at-most-one: forbid every pair simultaneously true."""
    return And([Not(And(a, b)) for a, b in combinations(list(bool_vars), 2)])


def exactly_one_np(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    """Exactly-one using naive pairwise encoding."""
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))


# ---------------- Binary (BW) encoding -------------------------------------

def at_least_one_bw(bool_vars: Sequence[BoolRef]) -> BoolRef:
    """Binary at-least-one reuses naive disjunction (keeps behaviour)."""
    return at_least_one_np(bool_vars)


def at_most_one_bw(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    """Binary encoding for at-most-one using binary index variables.

    This implementation mirrors the original logic: introduce m = ceil(log2(n))
    binary selector bits r_i and constrain each original variable to imply the
    corresponding selector assignment for its index.
    """
    n = len(bool_vars)
    if n == 0:
        return BoolVal(True)

    m = math.ceil(math.log2(n)) if n > 1 else 1
    r = [Bool(f"r_{name}_{i}") for i in range(m)]
    binaries = [to_binary(idx, m) for idx in range(n)]

    constraints = []
    for i, v in enumerate(bool_vars):
        bits = [r[j] if binaries[i][j] == "1" else Not(r[j]) for j in range(m)]
        constraints.append(Or(Not(v), And(*bits)))

    return And(constraints)


def exactly_one_bw(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    return And(at_least_one_bw(bool_vars), at_most_one_bw(bool_vars, name))


# ---------------- Sequential (SEQ) encoding for exactly-one -----------------

def at_least_one_seq(bool_vars: Sequence[BoolRef]) -> BoolRef:
    return at_least_one_np(bool_vars)


def at_most_one_seq(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    """Sequential encoding for at-most-one (specialized for exactly-one use)."""
    n = len(bool_vars)
    if n <= 1:
        return BoolVal(True)

    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints: List[BoolRef] = []

    # first and last positions
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n - 1]), Not(s[n - 2])))

    # middle positions
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1])))
        constraints.append(Or(Not(s[i - 1]), s[i]))

    return And(constraints)


def exactly_one_seq(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))


# ---------------- Heule encoding ------------------------------------------------

_heule_counter = 0


def heule_at_most_one(bool_vars: Sequence[BoolRef]) -> BoolRef:
    """Recursive Heule at-most-one encoding.

    Behaviour preserved from the original code. Uses a small base-case (<=4)
    where pairwise encoding is applied and otherwise introduces an auxiliary
    boolean and recurses.
    """
    global _heule_counter
    if len(bool_vars) <= 4:
        return And([Not(And(a, b)) for a, b in combinations(list(bool_vars), 2)])

    _heule_counter += 1
    aux = Bool(f"y_amo_{_heule_counter}")

    # split into first three vars + aux and remaining vars with negated aux
    left_part = at_most_one_np(list(bool_vars[:3]) + [aux])
    right_part = heule_at_most_one([Not(aux)] + list(bool_vars[3:]))
    return And(left_part, right_part)


def heule_exactly_one(bool_vars: Sequence[BoolRef], name: str = "") -> BoolRef:
    return And(heule_at_most_one(list(bool_vars)), at_least_one_np(list(bool_vars)))


# ---------------------------------------------------------------------------
# At-most-k encodings (general k)
# ---------------------------------------------------------------------------


def at_most_k_np(bool_vars: Sequence[BoolRef], k: int, name: str = "") -> BoolRef:
    """Naive direct encoding: forbid every (k+1)-combination to be true."""
    if k >= len(bool_vars):
        return BoolVal(True)
    if k < 0:
        return BoolVal(False)
    return And([Or([Not(x) for x in comb]) for comb in combinations(list(bool_vars), k + 1)])


def at_most_k_seq(bool_vars: Sequence[BoolRef], k: int, name: str = "") -> BoolRef:
    """Sequential counter encoding for at-most-k.

    Mirrors the original encoding while clarifying variable names.
    """
    n = len(bool_vars)
    if n == 0:
        return BoolVal(True)
    if k == 0:
        return And([Not(v) for v in bool_vars])
    if k >= n:
        return BoolVal(True)

    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(n - 1)]
    constraints: List[BoolRef] = []

    # initialize first row
    constraints.append(Or(Not(bool_vars[0]), s[0][0]))
    for j in range(1, k):
        constraints.append(Not(s[0][j]))

    for i in range(1, n - 1):
        constraints.append(Or(Not(s[i - 1][0]), s[i][0]))
        constraints.append(Or(Not(bool_vars[i]), s[i][0]))

        for j in range(1, k):
            constraints.append(Or(Not(s[i - 1][j]), s[i][j]))
            constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1][j - 1]), s[i][j]))

        constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1][k - 1])))

    constraints.append(Or(Not(bool_vars[n - 1]), Not(s[n - 2][k - 1])))

    return And(constraints)


# ---------------- Totalizer encoding ----------------------------------------


def totalizer_merge(left_sum: List[BoolRef], right_sum: List[BoolRef], name_prefix: str, depth: int, constraints: List[BoolRef]) -> List[BoolRef]:
    """Merge two partial sums for the totalizer encoding (side-effect: append constraints).

    The produced merged vector length is len(left) + len(right).
    """
    merged = [Bool(f"{name_prefix}_s_{depth}_{i}") for i in range(len(left_sum) + len(right_sum))]

    for i, lv in enumerate(left_sum):
        constraints.append(Implies(lv, merged[i]))
    for i, rv in enumerate(right_sum):
        constraints.append(Implies(rv, merged[i]))

    for i, lv in enumerate(left_sum):
        for j, rv in enumerate(right_sum):
            if i + j + 1 < len(merged):
                constraints.append(Implies(And(lv, rv), merged[i + j + 1]))

    return merged


def at_most_k_totalizer(bool_vars: Sequence[BoolRef], k: int, name: str = "") -> BoolRef:
    """Totalizer encoding for at-most-k constraint.

    Preserves the original tree-building logic and constraint generation.
    """
    n = len(bool_vars)
    if k >= n:
        return BoolVal(True)
    if k < 0:
        return BoolVal(False)
    if n == 0:
        return BoolVal(True)

    constraints: List[BoolRef] = []
    current_level: List[List[BoolRef]] = [[v] for v in bool_vars]
    depth = 0

    while len(current_level) > 1:
        next_level: List[List[BoolRef]] = []
        for i in range(0, len(current_level), 2):
            if i + 1 == len(current_level):
                next_level.append(current_level[i])
            else:
                left = current_level[i]
                right = current_level[i + 1]
                merged = totalizer_merge(left, right, name, depth, constraints)
                next_level.append(merged)
                depth += 1
        current_level = next_level

    total_sum = current_level[0]

    for i in range(k, len(total_sum)):
        constraints.append(Not(total_sum[i]))

    return And(constraints)


# ---------------------------------------------------------------------------
# Schedule utilities (matrix conversion, printing, circle matchings)
# ---------------------------------------------------------------------------


def convert_to_matrix(n: int, solution: Sequence[Tuple[int, int, int, int]]) -> List[List[Optional[List[int]]]]:
    """Convert solution tuples (home, away, week, period) into a matrix indexed
    by [period-1][week-1] storing [home, away]. Assumes 1-based values in the
    solution tuples (keeps original behaviour).
    """
    num_periods = n // 2
    num_weeks = n - 1
    matrix: List[List[Optional[List[int]]]] = [[None for _ in range(num_weeks)] for _ in range(num_periods)]
    for h, a, w, p in solution:
        matrix[p - 1][w - 1] = [h, a]
    return matrix


def save_results_as_json(n: int, results: Dict, model_name: str, output_dir: str = "./res/SAT") -> None:
    """Save solver results to a JSON file. Behaviour preserved from original code."""
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"{n}.json")

    existing = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    result_block: Dict[str, Dict] = {model_name: results}

    for method, res in result_block.items():
        runtime = res.get("time", 300.0)
        time_field = 300 if not res.get("optimal") else math.floor(runtime)
        sol = res.get("sol")
        matrix = convert_to_matrix(n, sol) if sol else []

        existing[method] = {
            "time": time_field,
            "optimal": res.get("optimal"),
            "obj": res.get("obj"),
            "sol": matrix,
        }

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=1)


def print_weekly_schedule(match_list: Optional[Sequence[Tuple[int, int, int, int]]], num_teams: int) -> None:
    """Pretty-print the weekly schedule. Preserves exact output format from original."""
    num_weeks = num_teams - 1
    num_periods = num_teams // 2

    print("\n--- Sport Tournament Scheduler ---")
    print(f"Number of Teams: {num_teams}")
    print(f"Number of Weeks: {num_weeks}")
    print(f"Periods per Week: {num_periods}")
    print("---------------------------\n")

    if match_list is None:
        print("No solution was found")
        return

    schedule: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for home_team, away_team, week, period in match_list:
        schedule[(week, period)] = (home_team, away_team)

    for w_idx in range(1, num_weeks + 1):
        print(f"Week {w_idx}:")
        for p_idx in range(1, num_periods + 1):
            match = schedule.get((w_idx, p_idx))
            if match:
                home_team, away_team = match
                print(f"  Period {p_idx}: Team {home_team} (Home) vs Team {away_team} (Away)")
            else:
                print(f"  Period {p_idx}: [No Scheduled Matches]")
        print()

    print("--- END SCHEDULE ---\n")


def circle_matchings(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """Circle method: generate fixed calendar matchings.

    Returns a dict mapping week (0-based) to a list of pairs (i, j) with 0-based
    team indices. Behaviour preserved from original implementation.
    """
    pivot = n - 1
    circle = list(range(n - 1))
    weeks = n - 1
    schedule: Dict[int, List[Tuple[int, int]]] = {}

    for w in range(weeks):
        matches = [(pivot, circle[w])]
        for k in range(1, n // 2):
            i = circle[(w + k) % (n - 1)]
            j = circle[(w - k + (n - 1)) % (n - 1)]
            matches.append((i, j))
        schedule[w] = matches

    return schedule


# ---------------------------------------------------------------------------
# Helper: lexicographical less-than for boolean vectors
# ---------------------------------------------------------------------------


def lex_less_bool(curr: Sequence[BoolRef], nxt: Sequence[BoolRef]) -> BoolRef:
    """Return a Z3 formula enforcing lexicographic curr < nxt over boolean vectors.

    Preserves original semantics: finds first position where curr is True and
    nxt False while prefixes are equal.
    """
    conditions: List[BoolRef] = []
    length = len(curr)
    for i in range(length):
        prefix_equal = [curr[j] == nxt[j] for j in range(i)]
        cond = And(*(prefix_equal + [curr[i], Not(nxt[i])]))
        conditions.append(cond)
    return Or(*conditions)


# ---------------------------------------------------------------------------
# Main STS modelling and solvers (create model, decisional, optimization)
# ---------------------------------------------------------------------------


def create_sts_model(
    n: int,
    max_diff_k: int,
    exactly_one_encoding: Callable[..., BoolRef],
    at_most_k_encoding: Callable[..., BoolRef],
    symmetry_breaking: bool = True,
) -> Tuple[Solver, Dict[Tuple[int, int, int], BoolRef], Dict[Tuple[int, int], BoolRef], Dict[Tuple[int, int], int]]:
    """Create Z3 model for STS with fixed calendar and PB bounds on home games.

    Returns (solver, match_period_vars, home_vars, pair_to_week).
    """
    if n % 2 != 0:
        raise ValueError("The number of teams must be even.")

    NUM_TEAMS = n
    NUM_WEEKS = n - 1
    NUM_PERIODS_PER_WEEK = n // 2

    solver = Solver()

    # Fixed calendar
    week_matchings = circle_matchings(NUM_TEAMS)
    pair_to_week: Dict[Tuple[int, int], int] = {}
    for w, matches in week_matchings.items():
        for i, j in matches:
            if i > j:
                i, j = j, i
            pair_to_week[(i, j)] = w

    # Boolean variables for matches in periods
    match_period_vars: Dict[Tuple[int, int, int], BoolRef] = {}
    for (i, j) in pair_to_week:
        for p in range(NUM_PERIODS_PER_WEEK):
            match_period_vars[(i, j, p)] = Bool(f"m_{i}_{j}_p{p}")

    # Home indicator variables (ordered pair keys i<j)
    home_vars: Dict[Tuple[int, int], BoolRef] = {}
    for i in range(NUM_TEAMS):
        for j in range(i + 1, NUM_TEAMS):
            home_vars[(i, j)] = Bool(f"home_{i}_{j}")

    # 1) Each match assigned to exactly one period
    for (i, j) in pair_to_week:
        vars_for_match = [match_period_vars[(i, j, p)] for p in range(NUM_PERIODS_PER_WEEK)]
        solver.add(exactly_one_encoding(vars_for_match, f"match_once_{i}_{j}"))

    # 2) Each period-slot in a week contains exactly one match
    for w in range(NUM_WEEKS):
        week_matches = week_matchings[w]
        for p in range(NUM_PERIODS_PER_WEEK):
            slot_vars = []
            for i, j in week_matches:
                if i > j:
                    i, j = j, i
                slot_vars.append(match_period_vars[(i, j, p)])
            solver.add(exactly_one_encoding(slot_vars, f"one_match_per_slot_w{w}_p{p}"))

    # 3) Each team appears at most twice in the same period across all weeks
    for t in range(NUM_TEAMS):
        for p in range(NUM_PERIODS_PER_WEEK):
            appearances: List[BoolRef] = []
            for (i, j), _w in pair_to_week.items():
                if t == i or t == j:
                    appearances.append(match_period_vars[(i, j, p)])
            solver.add(at_most_k_encoding(appearances, 2, f"team_{t}_max2_in_p{p}"))

    # Symmetry breaking
    if symmetry_breaking:
        # SB1: force a specific match to be in first period
        team_a, team_b = 0, NUM_TEAMS - 1
        solver.add(match_period_vars[(team_a, team_b, 0)])

        # SB2: enforce alternating home/away pattern for team 0 across weeks
        for (i, j), w in pair_to_week.items():
            if i == 0:
                solver.add(home_vars[(i, j)] if w % 2 == 0 else Not(home_vars[(i, j)]))
            elif j == 0:
                solver.add(Not(home_vars[(i, j)]) if w % 2 == 0 else home_vars[(i, j)])

        # SB3: lexicographic ordering of match-period vectors in week 0
        matches_week0 = sorted([(i, j) if i < j else (j, i) for (i, j) in week_matchings[0]])
        if len(matches_week0) > 1:
            bool_vectors = [[match_period_vars[(i, j, p)] for p in range(NUM_PERIODS_PER_WEEK)] for (i, j) in matches_week0]
            for a in range(len(bool_vectors) - 1):
                solver.add(lex_less_bool(bool_vectors[a], bool_vectors[a + 1]))

    # Optimization: bound home games per team by max_diff_k
    for t in range(NUM_TEAMS):
        home_game_terms: List[BoolRef] = []
        for (i, j), _w in pair_to_week.items():
            for p in range(NUM_PERIODS_PER_WEEK):
                mp = match_period_vars[(i, j, p)]
                if t == i:
                    home_game_terms.append(And(mp, home_vars[(i, j)]))
                elif t == j:
                    home_game_terms.append(And(mp, Not(home_vars[(i, j)])))

        NUM_GAMES = n - 1
        upper_bound = math.floor((NUM_GAMES + max_diff_k) / 2)
        lower_bound = math.ceil((NUM_GAMES - max_diff_k) / 2)

        solver.add(PbLe([(v, 1) for v in home_game_terms], upper_bound))
        solver.add(PbGe([(v, 1) for v in home_game_terms], lower_bound))

    return solver, match_period_vars, home_vars, pair_to_week


def _extract_schedule_from_model(model: ModelRef, match_period_vars: Dict[Tuple[int, int, int], BoolRef], home_vars: Dict[Tuple[int, int], BoolRef], pair_to_week: Dict[Tuple[int, int], int]) -> List[Tuple[int, int, int, int]]:
    """Given a model, extract the schedule as a list of 1-based tuples
    (home, away, week, period)."""
    schedule: List[Tuple[int, int, int, int]] = []
    for (i, j, p), var in match_period_vars.items():
        if is_true(model.evaluate(var)):
            key = (i, j) if i < j else (j, i)
            is_home = is_true(model.evaluate(home_vars[key]))

            if i < j:
                home_team_idx = i if is_home else j
                away_team_idx = j if is_home else i
            else:
                home_team_idx = i if is_home else j
                away_team_idx = j if is_home else i

            week_idx = pair_to_week[key]
            schedule.append((home_team_idx + 1, away_team_idx + 1, week_idx + 1, p + 1))
    return schedule


# ---------------------------------------------------------------------------
# Optimization solver (binary search on max_diff_k)
# ---------------------------------------------------------------------------


def solve_sts_optimization(
    n: int,
    timeout_seconds: int,
    exactly_one_encoding: Callable[..., BoolRef],
    at_most_k_encoding: Callable[..., BoolRef],
    symmetry_breaking: bool = True,
    verbose: bool = False,
) -> Dict:
    """Binary-search optimization on the maximum home/away imbalance (min-max).

    Behaviour and return format preserved exactly from the original.
    """
    if n % 2 != 0:
        raise ValueError("The number of teams must be even.")

    NUM_WEEKS = n - 1
    low, high = 1, NUM_WEEKS
    optimal_diff = None
    best_model: Optional[ModelRef] = None
    best_vars = None
    found_solution = False
    proven_unsat = False

    start_time = time.time()
    if verbose:
        print(f"\n--- Optimization for n={n} started ---")

    last_solver = None

    while low <= high:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 3:
            if verbose:
                print("Global timeout reached.")
            break

        k = (low + high) // 2
        if verbose:
            print(f"Testing max_diff <= {k}. Remaining time: {remaining:.2f}s...")

        solver, match_period_vars, home_vars, pair_to_week = create_sts_model(
            n=n,
            max_diff_k=k,
            exactly_one_encoding=exactly_one_encoding,
            at_most_k_encoding=at_most_k_encoding,
            symmetry_breaking=symmetry_breaking,
        )
        last_solver = solver
        solver.set("random_seed", 42)
        solver.set("timeout", int(remaining * 1000))

        status = solver.check()
        if verbose:
            print(f"  Solver result for k={k}: {status}")

        if status == sat:
            model = solver.model()
            optimal_diff = k
            best_model = model
            best_vars = (match_period_vars, home_vars, pair_to_week)
            found_solution = True
            high = k - 1
            if verbose and k > 1:
                print(f"  Found a solution with max_diff <= {k}. Trying smaller value.")

        elif status == unsat:
            proven_unsat = True
            break

        else:
            if verbose:
                print("  Solver returned 'unknown'.")
            break

    stats = last_solver.statistics() if last_solver is not None else {}
    final_stats = {
        'restarts': stats.get_key_value('restarts') if 'restarts' in stats.keys() else 0,
        'max_memory': stats.get_key_value('max memory') if 'max memory' in stats.keys() else 0,
        'mk_bool_var': stats.get_key_value('mk bool var') if 'mk bool var' in stats.keys() else 0,
        'conflicts': stats.get_key_value('conflicts') if 'conflicts' in stats.keys() else 0,
    }

    total_time = min(time.time() - start_time, timeout_seconds)

    best_schedule: List[Tuple[int, int, int, int]] = []
    proven_optimal_final = False
    if best_model is not None and best_vars is not None:
        match_period_vars, home_vars, pair_to_week = best_vars
        best_schedule = _extract_schedule_from_model(best_model, match_period_vars, home_vars, pair_to_week)
        if optimal_diff is not None and optimal_diff == 1:
            proven_optimal_final = True

    if proven_optimal_final:
        return {
            'obj': optimal_diff,
            'sol': best_schedule,
            'optimal': True,
            'time': total_time,
            'restart': final_stats['restarts'],
            'max_memory': final_stats['max_memory'],
            'mk_bool_var': final_stats['mk_bool_var'],
            'conflicts': final_stats['conflicts'],
        }
    if proven_unsat:
        return {
            'obj': None,
            'sol': best_schedule,
            'optimal': True,
            'time': total_time,
            'restart': final_stats['restarts'],
            'max_memory': final_stats['max_memory'],
            'mk_bool_var': final_stats['mk_bool_var'],
            'conflicts': final_stats['conflicts'],
        }
    if found_solution:
        return {
            'obj': optimal_diff,
            'sol': best_schedule,
            'optimal': False,
            'time': total_time,
            'restart': final_stats['restarts'],
            'max_memory': final_stats['max_memory'],
            'mk_bool_var': final_stats['mk_bool_var'],
            'conflicts': final_stats['conflicts'],
        }

    return {
        'obj': None,
        'sol': None,
        'optimal': False,
        'time': total_time,
        'restart': final_stats['restarts'],
        'max_memory': final_stats['max_memory'],
        'mk_bool_var': final_stats['mk_bool_var'],
        'conflicts': final_stats['conflicts'],
    }


# ---------------------------------------------------------------------------
# Decisional solver (find one solution for a given max_diff_k)
# ---------------------------------------------------------------------------


def solve_sts_decisional(
    n: int,
    max_diff_k: int,
    timeout_seconds: int,
    exactly_one_encoding: Callable[..., BoolRef],
    at_most_k_encoding: Callable[..., BoolRef],
    symmetry_breaking: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run a single SAT query for a fixed max_diff_k and return one solution if any."""
    if verbose:
        print(f"\n--- Decisional solver for n={n} ---")

    start_time = time.time()
    solver, match_period_vars, home_vars, pair_to_week = create_sts_model(
        n=n,
        max_diff_k=max_diff_k,
        exactly_one_encoding=exactly_one_encoding,
        at_most_k_encoding=at_most_k_encoding,
        symmetry_breaking=symmetry_breaking,
    )

    solver.set("random_seed", 42)
    solver.set("timeout", int(timeout_seconds * 1000))

    status = solver.check()
    solve_time = time.time() - start_time

    stats = solver.statistics()
    stats_dict = {
        'restarts': stats.get_key_value('restarts') if 'restarts' in stats.keys() else 0,
        'max_memory': stats.get_key_value('max memory') if 'max memory' in stats.keys() else 0,
        'mk_bool_var': stats.get_key_value('mk bool var') if 'mk bool var' in stats.keys() else 0,
        'conflicts': stats.get_key_value('conflicts') if 'conflicts' in stats.keys() else 0,
    }

    best_schedule: List[Tuple[int, int, int, int]] = []
    if status == sat:
        model = solver.model()
        best_schedule = _extract_schedule_from_model(model, match_period_vars, home_vars, pair_to_week)

    solve_time = solve_time if solve_time <= 300 else 300
    optimal = True if solve_time < 300 else False

    return {
        'obj': None,
        'sol': best_schedule,
        'optimal': optimal,
        'time': solve_time,
        'restart': stats_dict['restarts'],
        'max_memory': stats_dict['max_memory'],
        'mk_bool_var': stats_dict['mk_bool_var'],
        'conflicts': stats_dict['conflicts'],
    }


# ---------------------------------------------------------------------------
# CLI parsing and main()
# ---------------------------------------------------------------------------


def parse_n_teams(n_input: Iterable[str]) -> List[int]:
    """Parse -n values: accepts numbers and ranges (e.g. 2-18). Only even numbers kept."""
    values = set()
    for item in n_input:
        if re.match(r"^\d+-\d+$", item):
            start, end = map(int, item.split("-"))
            for n in range(start, end + 1):
                if n % 2 == 0:
                    values.add(n)
        else:
            try:
                n = int(item)
                if n % 2 == 0:
                    values.add(n)
                else:
                    print(f"[WARNING] Skipping odd number: {n}")
            except ValueError:
                print(f"[WARNING] Invalid value for -n: {item}")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sport Tournament Scheduler using Z3 solvers.")
    parser.add_argument("-n", "--n_teams", type=str, nargs='+', default=["2-20"],
                        help="List of even numbers or ranges like 2-18 for number of teams to test.")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Timeout in seconds for each solver instance.")
    parser.add_argument("--exactly_one_encoding", type=str, choices=["np", "bw", "seq", "heule"], help="Encoding for exactly-one constraints.")
    parser.add_argument("--at_most_k_encoding", type=str, choices=["np", "seq", "totalizer"], help="Encoding for at-most-k constraints.")
    parser.add_argument("--all", action="store_true", help="Run all combinations of encoding methods.")
    parser.add_argument("--run_decisional", action="store_true", help="Run the decisional solver.")
    parser.add_argument("--run_optimization", action="store_true", help="Run the optimization solver.")
    parser.add_argument("--sb", dest="sb", action="store_true", help="Enable symmetry breaking.")
    parser.add_argument("--no_sb", dest="sb", action="store_false", help="Disable symmetry breaking.")
    parser.set_defaults(sb=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--save_json", action='store_true', help="Save solver results to JSON files.")

    args = parser.parse_args()

    args.n_teams = parse_n_teams(args.n_teams)

    exactly_one_encodings = {
        "np": exactly_one_np,
        "bw": exactly_one_bw,
        "seq": exactly_one_seq,
        "heule": heule_exactly_one,
    }
    at_most_k_encodings = {
        "np": at_most_k_np,
        "seq": at_most_k_seq,
        "totalizer": at_most_k_totalizer,
    }

    if args.all:
        allowed_pairs = [
            ("np", "np"),
            ("heule", "seq"),
            ("heule", "totalizer"),
        ]
        encoding_combinations = [
            ((eo, exactly_one_encodings[eo]), (ak, at_most_k_encodings[ak]))
            for eo, ak in allowed_pairs
        ]
        if not args.run_decisional and not args.run_optimization:
            args.run_decisional = True
            args.run_optimization = True
    else:
        if not args.exactly_one_encoding or not args.at_most_k_encoding:
            print("Error: You must specify both --exactly_one_encoding and --at_most_k_encoding, or use --all.")
            return
        encoding_combinations = [(
            (args.exactly_one_encoding, exactly_one_encodings[args.exactly_one_encoding]),
            (args.at_most_k_encoding, at_most_k_encodings[args.at_most_k_encoding]),
        )]

    if not args.run_decisional and not args.run_optimization:
        print("Error: You must choose to run either --run_decisional or --run_optimization (or both).")
        parser.print_help()
        return

    timeout = args.timeout - 1
    sb_options = [True, False] if args.sb is None else [args.sb]

    for sb in sb_options:
        sb_name = "sb" if sb else "no_sb"
        for (eo_name, eo_func), (ak_name, ak_func) in encoding_combinations:
            name_prefix = f"{eo_name}_{ak_name}"

            if args.run_decisional:
                for n in args.n_teams:
                    model_name = f"decisional_{name_prefix}_{sb_name}"
                    try:
                        results = solve_sts_decisional(
                            n,
                            max_diff_k=n - 1,
                            exactly_one_encoding=eo_func,
                            at_most_k_encoding=ak_func,
                            timeout_seconds=timeout,
                            symmetry_breaking=sb,
                            verbose=args.verbose,
                        )
                    except ValueError as e:
                        print(f"Skipping n={n}: {e}")
                        continue

                    if args.save_json:
                        save_results_as_json(n, results=results, model_name=model_name)

                    if results['sol'] is not None:
                        if os.path.exists("/.dockerenv"):
                            os.system(f"echo '[Decisional Result] n={n} | time={results['time']}'")
                        else:
                            print(f"[Decisional Result] n={n} | time={results['time']}")
                    else:
                        print(f"[!] No solution found for n={n}")

            if args.run_optimization:
                for n in args.n_teams:
                    model_name = f"optimization_{name_prefix}_{sb_name}"
                    try:
                        results = solve_sts_optimization(
                            n,
                            timeout_seconds=timeout,
                            exactly_one_encoding=eo_func,
                            at_most_k_encoding=ak_func,
                            symmetry_breaking=sb,
                            verbose=args.verbose,
                        )
                    except ValueError as e:
                        print(f"Skipping n={n}: {e}")
                        continue

                    if args.save_json:
                        save_results_as_json(n, results=results, model_name=model_name)

                    if results['sol'] is not None:
                        if os.path.exists("/.dockerenv"):
                            os.system(f"echo '[Optimization Result] n={n} | obj={results['obj']} | time={results['time']}'")
                        else:
                            print(f"[Optimization Result] n={n} | obj={results['obj']} | time={results['time']}")
                    else:
                        print(f"[!] No solution found for n={n}")


if __name__ == "__main__":
    main()
