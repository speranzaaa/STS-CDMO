import subprocess
import json
import os
import math
import time

BASE_CP_DIR = os.path.join("source", "CP")

MODEL_DECISION = os.path.join(BASE_CP_DIR, "cp_model.mzn")
MODEL_OPT = os.path.join(BASE_CP_DIR, "cp_model_opt.mzn")
MODEL_DECISION_NOSB = os.path.join(BASE_CP_DIR, "cp_model_nosb.mzn")
MODEL_OPT_NOSB = os.path.join(BASE_CP_DIR, "cp_model_opt_nosb.mzn")

DZN_DIR = BASE_CP_DIR
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
    if not sol: return None
    try:
        all_teams = set()
        home_counts = {}
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
        val = 0
        for t in range(1, n + 1):
            val += abs(2 * home_counts.get(t, 0) - weeks)
        return val
    except:
        return None

def run_solver(exp, dzn_path):
    is_opt = "opt" in exp["name"]
   
    command = [
        "minizinc",
        "--solver", exp["solver"],
        "--time-limit", str(TIME_LIMIT_MS),
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
   
    if "----------" in stdout_data:
        blocks = stdout_data.split("----------")
        for b in blocks:
            b = b.strip()
            if not b: continue
            try:
                start = b.find('[')
                end = b.rfind(']') + 1
                if start != -1 and end != -1:
                    final_sol = json.loads(b[start:end])
            except:
                pass
    rpt_time = math.floor(actual_time)
    optimal = False
    if "==========" in stdout_data:
        optimal = True
    elif not is_opt and final_sol:
        optimal = True
    else:
        if final_sol and actual_time < (TIME_LIMIT_SEC - 1):
            optimal = True
    if rpt_time >= TIME_LIMIT_SEC:
        rpt_time = TIME_LIMIT_SEC
        optimal = False
    obj_val = calculate_obj(final_sol) if is_opt else None
    return {"time": rpt_time, "optimal": optimal, "obj": obj_val, "sol": final_sol}

if os.path.exists(MODEL_DECISION) and os.path.exists(MODEL_OPT) and os.path.exists(MODEL_DECISION_NOSB) and os.path.exists(MODEL_OPT_NOSB):
    for inst in INSTANCES:
        dzn = os.path.join(DZN_DIR, f"{inst}.dzn")
        json_path = os.path.join(OUTPUT_DIR, f"{inst}.json")
       
        if not os.path.exists(dzn): continue
       
        data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: data = json.load(f)
            except: pass
        for exp in EXPERIMENTS:
            if not os.path.exists(exp["model"]): continue
            data[exp["name"]] = run_solver(exp, dzn)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
