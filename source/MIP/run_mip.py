from pulp import *
import time
import argparse
import os
import json
import math

def get_variable_indexes(x):
    pairs_list = list(x.keys())
    first_pair = pairs_list[0]
    first_week_dict = x[first_pair]
    W = len(first_week_dict)
    first_week = next(iter(first_week_dict.values()))
    P = len(first_week)
    return len(pairs_list), W, P

def set_match_values(x, match_tuple, value, fix=False):
    for t in match_tuple:
        
        i=t[0]
        j=t[1]
        w=t[2]
        p=t[3]
        x[(i,j)][w][p].setInitialValue(value)
        if fix:
            x[(i,j)][w][p].fixValue()
    return x

def circle_method_pairs(n):
    assert n % 2 == 0 and n >= 4
    w, p = n - 1, n // 2
    fixed = 1
    others = list(range(2, n+1))
    schedule = []
    for wk in range(1, w+1):
        arr = [fixed] + others
        pairs = []
        for id_p in range(p):
            if (wk + id_p) % 2 == 0:
                pairs.append((arr[id_p], arr[-1 - id_p]))
            else:
                pairs.append((arr[-1 - id_p], arr[id_p]))
        schedule.append(pairs)
        others = [others[-1]] + others[:-1]
    return schedule

def format_matches(x):
    pairs_list, Ws, Ps = get_variable_indexes(x)
    schedule = []
    for p in range(1,Ps+1):
        new_period = []
        for w in range(1,Ws+1):
            for pair in x.keys():
                i, j = pair
                if type(x[pair][w][p].value())==float:
                    if x[pair][w][p].value()>0.5:
                        new_period.append([i,j])
        schedule.append(new_period)
    return schedule

def build_model(n, optimization, simmetry_br):
    games_per_pair = 1
    games_per_team_per_week = 1
    games_per_slot = 1
    max_games_per_team_per_period = 2

    TEAMS = range(1,n+1) # from 1 to n
    WEEKS = range(1,n) # from 1 to n-1
    PERIODS = range (1,(n//2)+1) # from 1 to n//2

    pairs=[(h,a) for h in TEAMS for a in TEAMS if h!=a]

    problem = LpProblem("Scheduling_Tournament", LpMinimize)

    #Decision variable
    matches = LpVariable.dicts("Match", (pairs, WEEKS, PERIODS), cat="Binary")

    #------ CONTRAINTS ------
    
    # 1. every team plays every other team once
    for (i,j) in pairs:
            if i<j:
                problem += lpSum(matches[(i,j)][w][p] + matches[(j,i)][w][p] for w in WEEKS for p in PERIODS) == games_per_pair, f"1c_{i}_{j}"

    # 2. every team plays exactly one game each week
    for t in TEAMS:
        for w in WEEKS:
            problem += lpSum(matches[(t,j)][w][p] + matches[(j,t)][w][p] for j in TEAMS if j!=t for p in PERIODS) == games_per_team_per_week, f"2c_{t}_{w}"

    # 3. every team plays at most 2 times in the same period
    for t in TEAMS:
        for p in PERIODS:
            problem += lpSum(matches[(t,j)][w][p] + matches[(j,t)][w][p] for j in TEAMS if j!=t for w in WEEKS) <= max_games_per_team_per_period, f"3c_{t}_{p}"

    # 4. each slot (week,period) has one game
    for w in WEEKS:
        for p in PERIODS:
            problem += lpSum(matches[(i,j)][w][p] for (i,j) in pairs) == games_per_slot, f"4c_{w}_{p}"


    #------ SIMMETRY BREAKINGS ------
    warm_time = 0
    if simmetry_br != "noSym":
        start_warm = time.time()

        if simmetry_br == 'fixWeek1' or simmetry_br == 'fixWeek2':
            fixed_matches = []
            neg_fixed_matches = []
            fixed_pairs = []
            for k in range(1,n,2):
                home = k
                away = k+1
                fixed_pairs.append((home, away))

            for period, (home_team, away_team) in enumerate(fixed_pairs, 1):
                fixed_matches.append((home_team, away_team, 1, period))
            
                for (i,j) in pairs:
                    if (i, j) != (home_team, away_team):
                        neg_fixed_matches.append((i, j, 1, period))
            
            set_match_values(matches, fixed_matches, 1, fix=True)
            set_match_values(matches, neg_fixed_matches, 0, fix=True)

        if simmetry_br == 'fixWeek2':
            fixed_matches = []
            neg_fixed_matches = []
            fixed_pairs = []

            util_list=[x for x in range(1,n+1) if x%2!=0] + [x for x in range(1,n+1) if x%2==0]
            for k in range(0,n,2):
                home = util_list[k]
                away = util_list[k+1]
                fixed_pairs.append((home, away))
            
            for period, (home_team, away_team) in enumerate(fixed_pairs, 1):
                fixed_matches.append((home_team, away_team, 2, period))
            
                for (i,j) in pairs:
                    if (i, j) != (home_team, away_team):
                        neg_fixed_matches.append((i, j, 2, period))
            
            set_match_values(matches, fixed_matches, 1, fix=True)
            set_match_values(matches, neg_fixed_matches, 0, fix=True)

        end_warm = time.time()
        warm_time = end_warm - start_warm

        if simmetry_br == "circleMethod":
            schedule = circle_method_pairs(n)
            pos_schedule = []
            neg_schedule = []
            for w, week in enumerate(schedule,1):
                for p, period in enumerate(week,1):
                    home = period[0]
                    away = period[1]
                    pos_schedule.append((home, away, w, p))
                    for (i,j) in pairs:
                        if (i,j) != (home,away):
                            neg_schedule.append((i, j, w, p))
            set_match_values(matches, pos_schedule, 1, fix=False)
            set_match_values(matches, neg_schedule, 0, fix=False)
                    


    #------ OPTIMIZATION ------
    if optimization != "feasible":
        if optimization == "homeAwayDiff1":
            home_away_diff = LpVariable.dicts("HomeAwayDiff", TEAMS, lowBound=1, upBound=n-1, cat="Integer")
            for t in TEAMS:
                home_games = lpSum(matches[(t,j)][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
                away_games = lpSum(matches[(j,t)][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
                
                problem += home_away_diff[t] >= home_games - away_games, f"diff1_{t}"
                problem += home_away_diff[t] >= away_games - home_games, f"diff2_{t}"
            
            problem += lpSum(home_away_diff[t] for t in TEAMS)

        if optimization == "homeAwayDiff0":
            home_away_diff = LpVariable.dicts("HomeAwayDiff", TEAMS, lowBound=0, upBound=n-2, cat="Integer")
            for t in TEAMS:
                home_games = lpSum(matches[(t,j)][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
                away_games = lpSum(matches[(j,t)][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
                
                problem += home_away_diff[t] +1>= home_games - away_games, f"diff1_{t}"
                problem += home_away_diff[t] +1>= away_games - home_games, f"diff2_{t}"
            
            problem += lpSum(home_away_diff[t] for t in TEAMS)

    return problem, matches, warm_time

def run_model(model, solver, matches):
    model.solve(solver)
    status = model.status
    # 0: 'No Solution Found', 1: 'Optimal Solution Found', 2: 'Solution Found', -1: 'No Solution Exists', -2: 'Solution is Unbounded'
    time = model.solutionCpuTime
    if status == 1 or status == 2:
        solution = matches
    else: 
        solution = None
    return status, time, solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run solver")
    parser.add_argument('--n', type=int, required=True, help='Number of teams')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit')
    parser.add_argument('--solver_type', type=str, default="cbc", choices=["cbc","highs","sciPy"], help='Solver type')
    parser.add_argument('--opt', type=str, default="feasible", choices=["homeAwayDiff0","homeAwayDiff1","feasible"], help='Optimization tecnique')
    parser.add_argument('--sym', type=str, default="noSym", choices=["circleMethod","fixWeek1","fixWeek2","noSym"], help='Symmetry breaking type')
    parser.add_argument('--json_path', type=str, default='', help='Path of the resulting json')
    parser.add_argument('--json_name', type=str, default='', help='Name of the resulting json')
    parser.add_argument('--solver_verbose', type=int, default=0, choices=[0,1], help='solver verbose type')

    args = parser.parse_args()
    n = args.n
    solver_type = args.solver_type
    opt = args.opt
    sym = args.sym
    if sym == "noSym":
        warmStart=False
    else:
        warmStart=True
    time_limit = args.time_limit
    solver_verbose = args.solver_verbose
    json_path = args.json_path
    json_name = args.json_name

    assert n>=2
    assert n%2==0

    #solution
    if solver_type == "cbc":
        solver = PULP_CBC_CMD(msg=solver_verbose, timeLimit=time_limit, warmStart=warmStart) 
    elif solver_type == "highs":
        solver = HiGHS(msg=solver_verbose, timeLimit=time_limit)
    elif solver_type=="sciPy":
        solver = SCIP_PY(msg=solver_verbose, timeLimit=time_limit, warmStart=warmStart)

    problem, matches, warm_time = build_model(n, opt, sym)
    status, solver_time, solution = run_model(problem, solver, matches)

    runtime = warm_time + solver_time
    runtime = int(math.floor(runtime))

    is_feasible = (status == 2 or status == 1) and runtime < time_limit

    is_optimal = (status == 1) and runtime < time_limit
    if opt == "feasible":
        is_optimal = False

    obj_value = None
    if opt != "feasible" and is_feasible:
        obj_value = int(problem.objective.value())
    
    if not is_feasible:
        runtime=300
        formatted_solution = []
    else:
        formatted_solution = format_matches(solution)
    
    
    if json_name.strip() == '':
        json_name = f"{n}.json"
    
    config_key = f"{solver_type}_{opt}_{sym}"
    
    if json_path.strip() != '':
        os.makedirs(json_path, exist_ok=True)
        full_path = os.path.join(json_path, json_name)

        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[config_key] = {
            "time": runtime,
            "optimal": is_optimal,
            "obj": obj_value,
            "sol": formatted_solution
        }
        
        # Write back to file
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)
            