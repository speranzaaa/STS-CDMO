import subprocess
import json
import os
import math
import time
import re

MODEL_DECISION = os.path.join("source", "CP", "cp_model.mzn")
MODEL_OPT = os.path.join("source", "CP", "cp_model_opt.mzn")
DZN_DIR = os.path.join("source", "CP")
OUTPUT_DIR = os.path.join("res", "CP")
INSTANCES = ["6", "8", "10", "12", "14", "16", "18"]
TIME_LIMIT_MS = 300000
TIME_LIMIT_SEC = 300

EXPERIMENTS = [
    {"name": "gecode", "solver": "gecode", "model": MODEL_DECISION},
    {"name": "chuffed", "solver": "chuffed", "model": MODEL_DECISION},
    {"name": "gecode_opt", "solver": "gecode", "model": MODEL_OPT}
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_objective_python(sol):
    if not sol: return None
    try:
        home_counts = {}
        all_teams = set()
        
        for period_row in sol:
            for match in period_row:
                if len(match) >= 2:
                    h, a = match[0], match[1]
                    all_teams.add(h)
                    all_teams.add(a)
                    home_counts[h] = home_counts.get(h, 0) + 1
                    if a not in home_counts: home_counts[a] = 0
        
        if not all_teams: return None
        n = max(all_teams)
        weeks = n - 1
        
        obj_val = 0
        for t in range(1, n + 1):
            hc = home_counts.get(t, 0)
            obj_val += abs(2 * hc - weeks)
        return obj_val
    except:
        return None

def parse_output_file(filepath, is_optimization):
    final_sol = []
    final_obj = None
    is_optimal = False
    
    if not os.path.exists(filepath):
        return [], None, False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if "==========" in content:
        is_optimal = True
    
    if not is_optimization and "----------" in content:
        is_optimal = True

    if "----------" in content:
        blocks = content.split("----------")
        for block in blocks:
            block = block.strip()
            if not block: continue
            try:
                start = block.find('[')
                end = block.rfind(']') + 1
                if start != -1 and end != -1:
                    sol_list = json.loads(block[start:end])
                    
                    current_obj = None
                    if is_optimization:
                        current_obj = calculate_objective_python(sol_list)
                    
                    final_sol = sol_list
                    final_obj = current_obj
            except:
                continue
                
    return final_sol, final_obj, is_optimal

def run_experiment(exp_config, dzn_path):
    model_path = exp_config["model"]
    solver_id = exp_config["solver"]
    is_optimization = "opt" in model_path or "opt" in exp_config["name"]
    
    temp_file = f"temp_{exp_config['name']}.txt"

    command = [
        "minizinc",
        "--solver", solver_id, 
        "--time-limit", str(TIME_LIMIT_MS), 
        model_path, dzn_path
    ]

    start_time = time.perf_counter()
    
    with open(temp_file, 'w', encoding='utf-8') as outfile:
        try:
            subprocess.run(command, stdout=outfile, stderr=subprocess.DEVNULL, timeout=TIME_LIMIT_SEC + 5)
        except subprocess.TimeoutExpired:
            pass 

    actual_runtime = time.perf_counter() - start_time
    
    sol, obj, optimal = parse_output_file(temp_file, is_optimization)
    
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass
    
    report_time = math.floor(actual_runtime)
    
    if report_time >= TIME_LIMIT_SEC:
        report_time = TIME_LIMIT_SEC
        if not sol:
            optimal = False
    else:
        if sol and actual_runtime < (TIME_LIMIT_SEC - 2):
            optimal = True

    if not is_optimization:
        obj = None

    return {"time": report_time, "optimal": optimal, "obj": obj, "sol": sol}

if os.path.exists(MODEL_DECISION) and os.path.exists(MODEL_OPT):
    for inst_name in INSTANCES:
        dzn_file = os.path.join(DZN_DIR, f"{inst_name}.dzn")
        json_file_path = os.path.join(OUTPUT_DIR, f"{inst_name}.json")
        
        if not os.path.exists(dzn_file):
            continue
            
        final_data = {}
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    final_data = json.load(f)
            except:
                final_data = {}

        for exp in EXPERIMENTS:
            if not os.path.exists(exp["model"]):
                continue
            result = run_experiment(exp, dzn_file)
            final_data[exp["name"]] = result

        with open(json_file_path, 'w') as f:
            json.dump(final_data, f, indent=4)
