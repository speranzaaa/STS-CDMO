#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from z3 import (
    And,
    Bool,
    BoolVal,
    Implies,
    Not,
    Or,
    PbGe,
    PbLe,
    Solver,
    is_true,
    sat,
    unsat,
)

# --------------------------------------------------------------
# Section: EXACTLY-ONE encodings
# --------------------------------------------------------------

# Naive / Pairwise encoding (NP)
def eo_at_least_one_pairwise(vars_: Sequence[Bool]) -> Bool:
    """At least one: simple OR of the variables."""
    return Or(list(vars_))


def eo_at_most_one_pairwise(vars_: Sequence[Bool], name: str = "") -> Bool:
    """At most one: pairwise forbids any pair being true at the same time."""
    return And([Not(And(a, b)) for a, b in combinations(vars_, 2)])


def eo_exactly_one_pairwise(vars_: Sequence[Bool], name: str = "") -> Bool:
    """Exactly one = at least one AND at most one (pairwise)."""
    return And(eo_at_least_one_pairwise(vars_), eo_at_most_one_pairwise(vars_, name))


# Binary / Log encoding (BW)
def _to_binary_str(num: int, length: Optional[int] = None) -> str:
    """Return binary representation (as string) padded to length if provided."""
    s = bin(num)[2:]
    if length is not None:
        return s.rjust(length, "0")
    return s


def eo_at_least_one_binary(vars_: Sequence[Bool]) -> Bool:
    """At least one with binary encoding just delegates to pairwise 'at least one'."""
    return eo_at_least_one_pairwise(vars_)


def eo_at_most_one_binary(vars_: Sequence[Bool], name: str = "") -> Bool:
    """
    At most one using a binary (log) encoding:
    - Introduces log2(n) auxiliary boolean variables r_i that encode the index.
    - For each original variable v_i, add clause: v_i -> (r matches binary(i))
    This enforces uniqueness when combined with a single 'at least one' over original vars.
    """
    n = len(vars_)
    if n == 0:
        return BoolVal(True)

    # number of bits needed to encode n items
    m = math.ceil(math.log2(max(1, n)))
    r = [Bool(f"r_{name}_{i}") for i in range(m)]
    binaries = [_to_binary_str(idx, m) for idx in range(n)]

    clauses = []
    for i in range(n):
        bits = []
        for j in range(m):
            bits.append(r[j] if binaries[i][j] == "1" else Not(r[j]))
        clauses.append(Or(Not(vars_[i]), And(*bits)))

    return And(clauses)


def eo_exactly_one_binary(vars_: Sequence[Bool], name: str = "") -> Bool:
    """Exactly one using binary encoding (at least one + binary at most one)."""
    return And(eo_at_least_one_binary(vars_), eo_at_most_one_binary(vars_, name))


# Sequential encoding (SEQ) for exactly-one (k = 1 case)
def eo_at_least_one_seq(vars_: Sequence[Bool]) -> Bool:
    """At least one using the pairwise at-least-one helper."""
    return eo_at_least_one_pairwise(vars_)


def eo_at_most_one_seq(vars_: Sequence[Bool], name: str = "") -> Bool:
    """
    Sequential encoding (Sinz) for at-most-one (specialized for exactly-one building).
    Uses chain variables s_i to represent prefix sums properties.
    """
    n = len(vars_)
    if n <= 1:
        return BoolVal(True)

    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    clauses = []
    # first and last boundary constraints
    clauses.append(Or(Not(vars_[0]), s[0]))
    clauses.append(Or(Not(vars_[n - 1]), Not(s[n - 2])))

    # middle constraints (propagate the sequential counter)
    for i in range(1, n - 1):
        clauses.append(Or(Not(vars_[i]), s[i]))
        clauses.append(Or(Not(vars_[i]), Not(s[i - 1])))
        clauses.append(Or(Not(s[i - 1]), s[i]))

    return And(clauses)


def eo_exactly_one_seq(vars_: Sequence[Bool], name: str = "") -> Bool:
    """Exactly one via sequential encoding (at least + at most)."""
    return And(eo_at_least_one_seq(vars_), eo_at_most_one_seq(vars_, name))


# Heule encoding (recursive hybrid)
_global_heule_counter = 0  # auxiliary counter for naming helper variables


def eo_heule_at_most_one(vars_: Sequence[Bool]) -> Bool:
    """
    Heule's at-most-one encoding:
    - If small (<= 4) use pairwise
    - Else, introduce auxiliary variable and recursively encode a split
    This mirrors the original recursive strategy.
    """
    if len(vars_) <= 4:
        return And([Not(And(a, b)) for a, b in combinations(vars_, 2)])
    else:
        global _global_heule_counter
        _global_heule_counter += 1
        aux = Bool(f"y_amo_{_global_heule_counter}")
        # Use original small prefix together with aux, and recursively handle rest
        return And(eo_at_most_one_pairwise(vars_[:3] + [aux]), eo_heule_at_most_one([Not(aux)] + vars_[3:]))


def eo_heule_exactly_one(vars_: Sequence[Bool], name: str = "") -> Bool:
    """Exactly-one with Heule at-most-one and simple at-least-one."""
    return And(eo_heule_at_most_one(vars_), eo_at_least_one_pairwise(vars_))


# --------------------------------------------------------------
# Section: AT-MOST-K encodings
# --------------------------------------------------------------

# Direct (NP) at-most-k: forbid every combination of size k+1
def amk_np(vars_: Sequence[Bool], k: int, name: str = "") -> Bool:
    """At-most-k by enumerating all (k+1)-subsets and forbidding them simultaneously."""
    if k >= len(vars_):
        return BoolVal(True)
    if k < 0:
        return BoolVal(False)
    return And([Or([Not(x) for x in comb]) for comb in combinations(vars_, k + 1)])


# Sequential counter encoding for general k
def amk_seq(vars_: Sequence[Bool], k: int, name: str = "") -> Bool:
    """
    Sequential counter encoding for at-most-k:
    - Builds a grid s[i][j] meaning: among prefix up to i, there are at least j+1 true variables.
    - Implements the Sinz-like sequential construction generalized to k.
    """
    n = len(vars_)
    if n == 0:
        return BoolVal(True)
    if k == 0:
        return And([Not(v) for v in vars_])
    if k >= n:
        return BoolVal(True)

    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(n - 1)]
    clauses = []

    # initialization for first row
    clauses.append(Or(Not(vars_[0]), s[0][0]))
    for j in range(1, k):
        clauses.append(Not(s[0][j]))

    # fill middle rows
    for i in range(1, n - 1):
        clauses.append(Or(Not(s[i - 1][0]), s[i][0]))
        clauses.append(Or(Not(vars_[i]), s[i][0]))

        for j in range(1, k):
            clauses.append(Or(Not(s[i - 1][j]), s[i][j]))
            clauses.append(Or(Not(vars_[i]), Not(s[i - 1][j - 1]), s[i][j]))

        clauses.append(Or(Not(vars_[i]), Not(s[i - 1][k - 1])))

    # final constraint for last variable
    clauses.append(Or(Not(vars_[n - 1]), Not(s[n - 2][k - 1])))

    return And(clauses)


# Totalizer encoding helpers
def _totalizer_merge(left: List[Bool], right: List[Bool], prefix: str, depth: int, constraints: List) -> List[Bool]:
    """
    Merge two partial 'sum' vectors in the totalizer tree.
    The merged vector has length len(left)+len(right) and constraints relate inputs to merged outputs.
    """
    merged = [Bool(f"{prefix}_s_{depth}_{i}") for i in range(len(left) + len(right))]

    # input items imply corresponding positions in merged vector
    for i in range(len(left)):
        constraints.append(Implies(left[i], merged[i]))
    for i in range(len(right)):
        constraints.append(Implies(right[i], merged[i]))

    # cross implications (left_i AND right_j) -> merged[i+j+1]
    for i in range(len(left)):
        for j in range(len(right)):
            idx = i + j + 1
            if idx < len(merged):
                constraints.append(Implies(And(left[i], right[j]), merged[idx]))

    return merged


def amk_totalizer(vars_: Sequence[Bool], k: int, name: str = "") -> Bool:
    """
    At-most-k implemented with a totalizer tree.
    Returns a Bool formula that enforces at most k true variables.
    """
    n = len(vars_)
    if k >= n:
        return BoolVal(True)
    if k < 0:
        return BoolVal(False)
    if n == 0:
        return BoolVal(True)

    constraints: List = []
    # start with leaves: each variable becomes a length-1 vector
    current_level: List[List[Bool]] = [[v] for v in vars_]
    depth = 0

    # build tree merging neighboring vectors
    while len(current_level) > 1:
        next_level: List[List[Bool]] = []
        for i in range(0, len(current_level), 2):
            if i + 1 == len(current_level):
                next_level.append(current_level[i])
            else:
                left = current_level[i]
                right = current_level[i + 1]
                merged = _totalizer_merge(left, right, name, depth, constraints)
                next_level.append(merged)
                depth += 1
        current_level = next_level

    total_sum = current_level[0]
    # enforce that positions >= k are false (i.e., at most k true)
    for idx in range(k, len(total_sum)):
        constraints.append(Not(total_sum[idx]))

    return And(constraints)


# --------------------------------------------------------------
# Utilities: convert, save, pretty-print
# --------------------------------------------------------------
def convert_to_matrix(n: int, solution: Sequence[Tuple[int, int, int, int]]) -> List[List[Optional[List[int]]]]:
    """
    Convert solution (list of tuples (home, away, week, period), 1-based)
    to a matrix indexed by [period-1][week-1] storing [home, away].
    """
    periods = n // 2
    weeks = n - 1
    matrix: List[List[Optional[List[int]]]] = [[None for _ in range(weeks)] for _ in range(periods)]
    for home, away, week, period in solution:
        matrix[period - 1][week - 1] = [home, away]
    return matrix


def save_results_as_json(n: int, results: Dict, model_name: str, output_dir: str = "/res/SAT") -> None:
    """
    Save the results for a given n into JSON file under output_dir.
    The function merges with existing JSON (if present).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{n}.json"

    data = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            data = {}

    # keep original interface: model_name -> results
    entry = {model_name: results}

    for method, res in entry.items():
        runtime = res.get("time", 300.0)
        time_field = 300 if not res.get("optimal") else math.floor(runtime)
        sol = res.get("sol")
        matrix = convert_to_matrix(n, sol) if sol else []
        data[method] = {
            "time": time_field,
            "optimal": res.get("optimal"),
            "obj": res.get("obj"),
            "sol": matrix,
        }

    json_path.write_text(json.dumps(data, indent=1))


def print_weekly_schedule(match_list: Sequence[Tuple[int, int, int, int]], num_teams: int) -> None:
    """
    Human-friendly printout of a schedule where match_list contains
    (home, away, week, period) with 1-based numbers.
    """
    weeks = num_teams - 1
    periods = num_teams // 2

    print("\n--- Sport Tournament Scheduler ---")
    print(f"Number of Teams: {num_teams}")
    print(f"Number of Weeks: {weeks}")
    print(f"Periods per Week: {periods}")
    print("---------------------------\n")

    if not match_list:
        print("No solution was found")
        return

    schedule = {(week, period): (home, away) for home, away, week, period in match_list}

    for w in range(1, weeks + 1):
        print(f"Week {w}:")
        for p in range(1, periods + 1):
            if (w, p) in schedule:
                home, away = schedule[(w, p)]
                print(f"  Period {p}: Team {home} (Home) vs Team {away} (Away)")
            else:
                print(f"  Period {p}: [No Scheduled Matches]")
        print()

    print("--- END SCHEDULE ---\n")


# --------------------------------------------------------------
# Circle method: fixed round-robin calendar
# --------------------------------------------------------------
def circle_matchings(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """
    Create a fixed calendar using the classical 'circle method'.
    Returns a mapping week_index (0-based) -> list of pairs (team_i, team_j) with 0-based indices.
    """
    pivot = n - 1
    circle = list(range(n - 1))
    weeks = n - 1
    mapping: Dict[int, List[Tuple[int, int]]] = {}

    for w in range(weeks):
        matches = [(pivot, circle[w])]
        for k in range(1, n // 2):
            i = circle[(w + k) % (n - 1)]
            j = circle[(w - k + (n - 1)) % (n - 1)]
            matches.append((i, j))
        mapping[w] = matches

    return mapping


# --------------------------------------------------------------
# Misc helper: lexicographical ordering for boolean vectors
# --------------------------------------------------------------
def lex_less_bool(curr: Sequence[Bool], nxt: Sequence[Bool]) -> Bool:
    """
    Return a Boolean formula asserting curr < nxt in lexicographic order.
    Implementation: OR over positions i of (curr[0]==nxt[0] & ... & curr[i-1]==nxt[i-1] & curr[i] & not nxt[i]).
    """
    conditions = []
    for i in range(len(curr)):
        prefix_eq = [curr[j] == nxt[j] for j in range(i)]
        conditions.append(And(*(prefix_eq + [curr[i], Not(nxt[i])])))
    return Or(*conditions)


# --------------------------------------------------------------
# Model builder: create STS Z3 model for given parameters
# --------------------------------------------------------------
def create_sts_model(
    n: int,
    max_diff_k: int,
    exactly_one_encoding: Callable[[Sequence[Bool], str], Bool],
    at_most_k_encoding: Callable[[Sequence[Bool], int, str], Bool],
    symmetry_breaking: bool = True,
):
    """
    Build a Z3 solver and variables for the Sports Tournament Scheduling (STS) problem.
    Returns (solver, match_period_vars, home_vars, pair_to_week).
    - match_period_vars: dict keyed by (i, j, p) -> Bool indicating match (i,j) assigned to period p
    - home_vars: dict keyed by (i, j) with i<j -> Bool meaning 'i is home when i vs j'
    - pair_to_week: mapping (i,j) -> week_index (0-based from circle_method)
    """
    if n % 2 != 0:
        raise ValueError("Number of teams must be even")

    num_teams = n
    num_weeks = n - 1
    periods_per_week = n // 2

    solver = Solver()
    week_matchings = circle_matchings(num_teams)
    pair_to_week: Dict[Tuple[int, int], int] = {}

    # canonicalize pairs as (small, large) to be consistent
    for w, matches in week_matchings.items():
        for (i, j) in matches:
            a, b = (i, j) if i < j else (j, i)
            pair_to_week[(a, b)] = w

    # match_period_vars: for each unordered pair (i, j) and each period p
    match_period_vars: Dict[Tuple[int, int, int], Bool] = {}
    for (i, j) in pair_to_week:
        for p in range(periods_per_week):
            match_period_vars[(i, j, p)] = Bool(f"m_{i}_{j}_p{p}")

    # home decision vars for unordered pairs (i < j)
    home_vars: Dict[Tuple[int, int], Bool] = {}
    for i in range(num_teams):
        for j in range(i + 1, num_teams):
            home_vars[(i, j)] = Bool(f"home_{i}_{j}")

    # 1) Each match assigned to exactly one period (using chosen encoding)
    for (i, j) in pair_to_week:
        vars_for_match = [match_period_vars[(i, j, p)] for p in range(periods_per_week)]
        solver.add(exactly_one_encoding(vars_for_match, f"match_once_{i}_{j}"))

    # 2) Each period in each week contains exactly one match (slot constraint)
    for w in range(num_weeks):
        matches = week_matchings[w]
        for p in range(periods_per_week):
            slot_vars = []
            for (i, j) in matches:
                a, b = (i, j) if i < j else (j, i)
                slot_vars.append(match_period_vars[(a, b, p)])
            solver.add(exactly_one_encoding(slot_vars, f"slot_w{w}_p{p}"))

    # 3) Each team appears at most twice in the same period across the tournament
    for t in range(num_teams):
        for p in range(periods_per_week):
            occurrences: List[Bool] = []
            for (i, j), _w in pair_to_week.items():
                if t == i or t == j:
                    occurrences.append(match_period_vars[(i, j, p)])
            solver.add(at_most_k_encoding(occurrences, 2, f"team_{t}_max2_p{p}"))

    # Symmetry breaking (optional)
    if symmetry_breaking:
        # SB1: force specific match to be in first period (fixing one degree of freedom)
        team_a, team_b = 0, num_teams - 1
        solver.add(match_period_vars[(team_a, team_b, 0)])

        # SB2: fix home/away alternation for team 0 across weeks (consistent pattern)
        for (i, j), w in pair_to_week.items():
            if i == 0:
                solver.add(home_vars[(i, j)] if (w % 2 == 0) else Not(home_vars[(i, j)]))
            elif j == 0:
                solver.add(Not(home_vars[(i, j)]) if (w % 2 == 0) else home_vars[(i, j)])

        # SB3: lexicographic ordering of the boolean period vectors in week 0
        matches_week0 = sorted([(i, j) if i < j else (j, i) for (i, j) in week_matchings[0]])
        if len(matches_week0) > 1:
            vectors = [[match_period_vars[(i, j, p)] for p in range(periods_per_week)] for (i, j) in matches_week0]
            for idx in range(len(vectors) - 1):
                solver.add(lex_less_bool(vectors[idx], vectors[idx + 1]))

    # Optimization-style constraints: ensure each team's home games are within max_diff_k
    for t in range(num_teams):
        home_games: List[Bool] = []
        for (i, j), _w in pair_to_week.items():
            for p in range(periods_per_week):
                mp = match_period_vars[(i, j, p)]
                # if team t is the smaller index i, 'home' means home_vars[(i,j)] true
                if t == i:
                    home_games.append(And(mp, home_vars[(i, j)]))
                elif t == j:
                    # if team t is the larger index j, home if home_vars[(i,j)] is false
                    home_games.append(And(mp, Not(home_vars[(i, j)])))

        num_games = n - 1
        upper = math.floor((num_games + max_diff_k) / 2)
        lower = math.ceil((num_games - max_diff_k) / 2)

        # PbLe / PbGe constraints over boolean indicators
        solver.add(PbLe([(v, 1) for v in home_games], upper))
        solver.add(PbGe([(v, 1) for v in home_games], lower))

    return solver, match_period_vars, home_vars, pair_to_week


# --------------------------------------------------------------
# Solver runners: optimization (binary search) and decisional
# --------------------------------------------------------------
def solve_sts_optimization(
    n: int,
    timeout_seconds: int,
    exactly_one_encoding: Callable[[Sequence[Bool], str], Bool],
    at_most_k_encoding: Callable[[Sequence[Bool], int, str], Bool],
    symmetry_breaking: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Optimization driver: binary-search on max_diff_k to minimize maximum home/away imbalance.
    Returns a result dictionary with keys: obj, sol, optimal, time, restart, max_memory, mk_bool_var, conflicts
    """
    if n % 2 != 0:
        raise ValueError("Number of teams must be even")

    num_weeks = n - 1
    low, high = 1, num_weeks
    best_model = None
    best_vars = None
    best_obj = None
    found_solution = False
    proven_unsat = False

    start = time.time()
    if verbose:
        print(f"\n--- Optimization for n={n} started ---")

    last_solver = None

    while low <= high:
        elapsed = time.time() - start
        remaining = timeout_seconds - elapsed
        if remaining <= 3:
            if verbose:
                print("Global timeout reached.")
            break

        k = (low + high) // 2
        if verbose:
            print(f"Testing max_diff <= {k}. Remaining: {remaining:.2f}s")

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
            print(f"  Solver returned: {status}")

        if status == sat:
            model = solver.model()
            best_model = model
            best_vars = (match_period_vars, home_vars, pair_to_week)
            best_obj = k
            found_solution = True
            high = k - 1
            if verbose and k > 1:
                print(f"  Found feasible with max_diff <= {k}, trying smaller")
        elif status == unsat:
            proven_unsat = True
            break
        else:
            # unknown or timed out in solver
            if verbose:
                print("  Solver returned 'unknown' or timed out.")
            break

    # gather final statistics from the last solver if present
    stats = last_solver.statistics() if last_solver is not None else None
    final_stats = {
        "restarts": stats.get_key_value("restarts") if stats and "restarts" in stats.keys() else 0,
        "max_memory": stats.get_key_value("max memory") if stats and "max memory" in stats.keys() else 0,
        "mk_bool_var": stats.get_key_value("mk bool var") if stats and "mk bool var" in stats.keys() else 0,
        "conflicts": stats.get_key_value("conflicts") if stats and "conflicts" in stats.keys() else 0,
    }

    total_time = time.time() - start
    total_time = min(total_time, timeout_seconds)

    schedule: List[Tuple[int, int, int, int]] = []
    proven_optimal = False

    if best_model is not None and best_vars is not None:
        mp_vars, home_vars, pair_to_week = best_vars
        for (i, j, p), var in mp_vars.items():
            if is_true(best_model.evaluate(var)):
                key = (i, j) if i < j else (j, i)
                is_home = is_true(best_model.evaluate(home_vars[key]))
                # Deduce home/away indices according to is_home flag
                if i < j:
                    home_idx = i if is_home else j
                    away_idx = j if is_home else i
                else:
                    home_idx = i if is_home else j
                    away_idx = j if is_home else i
                week_idx = pair_to_week[key]
                schedule.append((home_idx + 1, away_idx + 1, week_idx + 1, p + 1))

        # if best_obj == 1 we treat it as proven optimal final 
        if best_obj is not None and best_obj == 1:
            proven_optimal = True

    if proven_optimal:
        return {
            "obj": best_obj,
            "sol": schedule,
            "optimal": True,
            "time": total_time,
            **final_stats,
        }
    if proven_unsat:
        return {
            "obj": None,
            "sol": schedule,
            "optimal": True,
            "time": total_time,
            **final_stats,
        }
    if found_solution:
        return {
            "obj": best_obj,
            "sol": schedule,
            "optimal": False,
            "time": total_time,
            **final_stats,
        }
    return {
        "obj": None,
        "sol": None,
        "optimal": False,
        "time": total_time,
        **final_stats,
    }


def solve_sts_decisional(
    n: int,
    max_diff_k: int,
    timeout_seconds: int,
    exactly_one_encoding: Callable[[Sequence[Bool], str], Bool],
    at_most_k_encoding: Callable[[Sequence[Bool], int, str], Bool],
    symmetry_breaking: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Decisional solver: try to find a single feasible schedule with provided max_diff_k.
    Returns a result dict with the same structure as optimization runner (obj remains None).
    """
    if verbose:
        print(f"\n--- Decisional solver for n={n} (k={max_diff_k}) ---")

    start = time.time()
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
    elapsed = time.time() - start

    stats = solver.statistics()
    stats_dict = {
        "restarts": stats.get_key_value("restarts") if "restarts" in stats.keys() else 0,
        "max_memory": stats.get_key_value("max memory") if "max memory" in stats.keys() else 0,
        "mk_bool_var": stats.get_key_value("mk bool var") if "mk bool var" in stats.keys() else 0,
        "conflicts": stats.get_key_value("conflicts") if "conflicts" in stats.keys() else 0,
    }

    schedule: List[Tuple[int, int, int, int]] = []
    if status == sat:
        model = solver.model()
        for (i, j, p), var in match_period_vars.items():
            if is_true(model.evaluate(var)):
                key = (i, j) if i < j else (j, i)
                is_home = is_true(model.evaluate(home_vars[key]))
                if i < j:
                    home_idx = i if is_home else j
                    away_idx = j if is_home else i
                else:
                    home_idx = i if is_home else j
                    away_idx = j if is_home else i
                week_idx = pair_to_week[key]
                schedule.append((home_idx + 1, away_idx + 1, week_idx + 1, p + 1))

    # limit solve time to 300
    elapsed = elapsed if elapsed <= 300 else 300
    optimal_flag = True if elapsed < 300 else False

    return {
        "obj": None,
        "sol": schedule,
        "optimal": optimal_flag,
        "time": elapsed,
        **stats_dict,
    }


# --------------------------------------------------------------
# CLI helpers: parsing and main
# --------------------------------------------------------------
def parse_n_teams(inputs: Sequence[str]) -> List[int]:
    """
    Parse -n input values accepting numbers and ranges (e.g., "2-18").
    Only even numbers are returned (skipping odd).
    """
    values = set()
    for token in inputs:
        if re.match(r"^\d+-\d+$", token):
            a, b = map(int, token.split("-"))
            for x in range(a, b + 1):
                if x % 2 == 0:
                    values.add(x)
        else:
            try:
                v = int(token)
                if v % 2 == 0:
                    values.add(v)
                else:
                    print(f"[WARNING] Skipping odd number: {v}")
            except ValueError:
                print(f"[WARNING] Invalid -n token: {token}")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sport Tournament Scheduler using Z3 encodings.")
    parser.add_argument("-n", "--n_teams", type=str, nargs="+", default=["2-20"],
                        help="Even numbers or ranges like 2-18 for the number of teams.")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Timeout (s) per solver instance.")
    parser.add_argument("--exactly_one_encoding", type=str, choices=["np", "bw", "seq", "heule"],
                        help="Encoding for exactly-one.")
    parser.add_argument("--at_most_k_encoding", type=str, choices=["np", "seq", "totalizer"],
                        help="Encoding for at-most-k.")
    parser.add_argument("--all", action="store_true", help="Run a preset set of encoding combinations.")
    parser.add_argument("--run_decisional", action="store_true", help="Run decisional solver.")
    parser.add_argument("--run_optimization", action="store_true", help="Run optimization solver.")
    parser.add_argument("--sb", dest="sb", action="store_true", help="Enable symmetry breaking.")
    parser.add_argument("--no_sb", dest="sb", action="store_false", help="Disable symmetry breaking.")
    parser.set_defaults(sb=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--save_json", action="store_true", help="Save results to JSON files.")

    args = parser.parse_args()

    args.n_teams = parse_n_teams(args.n_teams)

    eo_map = {
        "np": eo_exactly_one_pairwise,
        "bw": eo_exactly_one_binary,
        "seq": eo_exactly_one_seq,
        "heule": eo_heule_exactly_one,
    }
    amk_map = {
        "np": amk_np,
        "seq": amk_seq,
        "totalizer": amk_totalizer,
    }

    # if --all, use a curated list of allowed combinations 
    if args.all:
        allowed = [("np", "np"), ("heule", "seq"), ("heule", "totalizer")]
        encoding_combinations = [((eo, eo_map[eo]), (ak, amk_map[ak])) for eo, ak in allowed]
        if not args.run_decisional and not args.run_optimization:
            args.run_decisional = True
            args.run_optimization = True
    else:
        if not args.exactly_one_encoding or not args.at_most_k_encoding:
            print("Error: specify both --exactly_one_encoding and --at_most_k_encoding, or use --all.")
            return
        encoding_combinations = [(
            (args.exactly_one_encoding, eo_map[args.exactly_one_encoding]),
            (args.at_most_k_encoding, amk_map[args.at_most_k_encoding]),
        )]

    if not args.run_decisional and not args.run_optimization:
        print("Error: choose at least --run_decisional or --run_optimization.")
        parser.print_help()
        return

    # subtract 1 from global timeout to leave a small margin 
    timeout = args.timeout - 1
    sb_options = [True, False] if args.sb is None else [args.sb]

    for sb in sb_options:
        sb_label = "sb" if sb else "no_sb"
        for (eo_name, eo_func), (ak_name, ak_func) in encoding_combinations:
            combo_name = f"{eo_name}_{ak_name}"

            if args.run_decisional:
                for n in args.n_teams:
                    model_name = f"decisional_{combo_name}_{sb_label}"
                    try:
                        results = solve_sts_decisional(
                            n=n,
                            max_diff_k=n - 1,
                            timeout_seconds=timeout,
                            exactly_one_encoding=eo_func,
                            at_most_k_encoding=ak_func,
                            symmetry_breaking=sb,
                            verbose=args.verbose,
                        )
                    except ValueError as exc:
                        print(f"Skipping n={n}: {exc}")
                        continue

                    if args.save_json:
                        save_results_as_json(n=n, results=results, model_name=model_name)

                    if results["sol"] is not None:
                        if os.path.exists("/.dockerenv"):
                            os.system(f"echo '[Decisional Result] n={n} | time={results['time']}'")
                        else:
                            print(f"[Decisional Result] n={n} | time={results['time']}")
                    else:
                        print(f"[!] No solution found for n={n}")

            if args.run_optimization:
                for n in args.n_teams:
                    model_name = f"optimization_{combo_name}_{sb_label}"
                    try:
                        results = solve_sts_optimization(
                            n=n,
                            timeout_seconds=timeout,
                            exactly_one_encoding=eo_func,
                            at_most_k_encoding=ak_func,
                            symmetry_breaking=sb,
                            verbose=args.verbose,
                        )
                    except ValueError as exc:
                        print(f"Skipping n={n}: {exc}")
                        continue

                    if args.save_json:
                        save_results_as_json(n=n, results=results, model_name=model_name)

                    if results["sol"] is not None:
                        if os.path.exists("/.dockerenv"):
                            os.system(f"echo '[Optimization Result] n={n} | obj={results['obj']} | time={results['time']}'")
                        else:
                            print(f"[Optimization Result] n={n} | obj={results['obj']} | time={results['time']}")
                    else:
                        print(f"[!] No solution found for n={n}")


if __name__ == "__main__":
    main()
