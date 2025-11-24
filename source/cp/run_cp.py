import subprocess
import json
import os
import math
import time
import sys

BASE_PATH = os.path.join("source", "CP")
if not os.path.exists(BASE_PATH):
    if os.path.exists("cp_model.mzn"):
        BASE_PATH = "."

MODEL_DECISION = os.path.join(BASE_PATH, "cp_model.mzn")
MODEL_OPT = os.path.join(BASE_PATH, "cp_model_opt.mzn")
MODEL_DECISION_NOSB = os.path.join(BASE_PATH, "cp_model_nosb.mzn")
MODEL_OPT_NOSB = os.path.join(BASE_PATH, "cp_model_opt_nosb.mzn")
DZN_DIR = BASE_PATH
OUTPUT_DIR = os.path.join("res", "CP")

INSTANCES = ["6", "8", "10", "12", "14", "16", "18"]
TIME_LIMIT_MS = 300000
TIME_LIMIT_SEC = 300
EXPERIMENTS = [
    {"name": "gecode", "solver": "gecode", "model": MODEL_DECISION},
    {"name": "chuffed", "solver": "chuffed", "model": MODEL_DECISION},
    {"name": "gecode_opt", "solver": "gecode", "model": MODEL_OPT},
    {"name": "gecode_without_sb", "solver": "gecode", "model": MODEL_DECISION_NOSB},
    {"name": "chuffed_without_sb", "solver": "chuffed", "model": MODEL_DECISION_NOSB},
    {"name": "gecode_opt_without_sb", "solver": "gecode", "model": MODEL_OPT_NOSB}
]
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_obj(sol):
    if not sol or not isinstance(sol, list): return None
    try:
        if len(sol) > 0 and not isinstance(sol[0], list): return None
        
        all_teams = set()
        home_counts = {}
        for period_row in sol:
            for match in period_row:
                if isinstance(match, list) and len(match) >= 2:
                    h, a = match[0], match[1]
                    all_teams.add(h)
                    all_teams.add(a)
                    home_counts[h] = home_counts.get(h, 0) + 1
                    if a not in home_counts: home_counts[a] = 0
       
        if not all_teams: return None
        n = max(all_teams)
        weeks = n - 1
        val = 0
        for t in range(1, n + 1):
            val += abs(2 * home_counts.get(t, 0) - weeks)
        return val
    except Exception:
        return None

def run_solver(exp, dzn_path):
    is_opt = "opt" in exp["name"]
    
    command = [
        "minizinc",
        "--solver", exp["solver"],
        "--time-limit", str(TIME_LIMIT_MS),
        "--output-mode", "json",
        exp["model"],
        dzn_path
    ]
    start_t = time.perf_counter()
    stdout_data = ""
    
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC + 10,
            encoding='utf-8'
        )
        stdout_data = res.stdout
        actual_time = time.perf_counter() - start_t
    except subprocess.TimeoutExpired as e:
        stdout_data = e.stdout if e.stdout else ""
        actual_time = TIME_LIMIT_SEC

    final_sol = []
    optimal = False
    
    blocks = stdout_data.split("----------")
    
    for block in blocks:
        block = block.strip()
        if not block: continue
        
        try:
            data = json.loads(block)
            
            if isinstance(data, dict):
                if "games" in data:
                    final_sol = data["games"]
                elif "schedule" in data:
                    final_sol = data["schedule"]
                elif "type" in data and data["type"] == "solution":
                     inner_data = data.get("data", {})
                     if "games" in inner_data:
                         final_sol = inner_data["games"]
                     elif "schedule" in inner_data:
                         final_sol = inner_data["schedule"]

            if final_sol:
                pass 
                
        except json.JSONDecodeError:
            continue

    if "==========" in stdout_data:
        optimal = True
    elif not is_opt and final_sol:
        optimal = True
    elif final_sol and actual_time < (TIME_LIMIT_SEC - 1):
        optimal = True
            
    rpt_time = math.floor(actual_time)
    if rpt_time >= TIME_LIMIT_SEC:
        rpt_time = TIME_LIMIT_SEC
        optimal = False

    obj_val = calculate_obj(final_sol) if is_opt else None
    
    return {"time": rpt_time, "optimal": optimal, "obj": obj_val, "sol": final_sol}

if __name__ == "__main__":
    
    if not os.path.exists(MODEL_DECISION):
        sys.exit(1)

    for inst in INSTANCES:
        dzn = os.path.join(DZN_DIR, f"{inst}.dzn")
        json_path = os.path.join(OUTPUT_DIR, f"{inst}.json")
        
        if not os.path.exists(dzn): continue
        
        data = {}

        for exp in EXPERIMENTS:
            if not os.path.exists(exp["model"]): continue

            result = run_solver(exp, dzn)
            data[exp["name"]] = result

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
