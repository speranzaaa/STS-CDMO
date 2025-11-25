import subprocess
import json
import os
import math
import time
import sys
import re
import argparse

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

ALL_EXPERIMENTS = [
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

def run_solver(exp, dzn_path, timeout_sec):
    is_opt = "opt" in exp["name"]
    timeout_ms = str(timeout_sec * 1000)
    
    command = [
        "minizinc",
        "--solver", exp["solver"],
        "--time-limit", timeout_ms,
        "--output-mode", "json",
        exp["model"],
        dzn_path
    ]
    
    start_t = time.perf_counter()
    stdout_data = ""
    stderr_data = ""
    
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False
        )
        stdout_data = res.stdout
        stderr_data = res.stderr
        actual_time = time.perf_counter() - start_t
    except Exception as e:
        stderr_data = str(e)
        actual_time = timeout_sec

    final_sol = []
    optimal = False
    
    json_candidates = []
    raw_blocks = re.findall(r'\{.*?\}', stdout_data, re.DOTALL)
    
    for block in raw_blocks:
        try:
            data = json.loads(block)
            json_candidates.append(data)
        except:
            pass

    for data in json_candidates:
        if not isinstance(data, dict): continue
        current_sol = []
        
        if "games" in data:
            current_sol = data["games"]
        elif "schedule" in data:
            current_sol = data["schedule"]
        elif "type" in data and data["type"] == "solution":
            inner = data.get("data", {})
            if "games" in inner:
                current_sol = inner["games"]
            elif "schedule" in inner:
                current_sol = inner["schedule"]
        
        if current_sol:
            final_sol = current_sol

    if "==========" in stdout_data:
        optimal = True
    elif not is_opt and final_sol:
        optimal = True
    elif final_sol and actual_time < (timeout_sec - 1):
        optimal = True
            
    rpt_time = math.floor(actual_time)
    if rpt_time >= timeout_sec:
        rpt_time = timeout_sec
        optimal = False

    obj_val = calculate_obj(final_sol) if is_opt else None

    if not final_sol and "UNSATISFIABLE" not in stdout_data:
        if stderr_data and len(stderr_data.strip()) > 0:
            print(f"[DEBUG] {exp['name']} Error: {stderr_data.strip()}")

    return {"time": rpt_time, "optimal": optimal, "obj": obj_val, "sol": final_sol}

def main():
    parser = argparse.ArgumentParser(description="Run CP Experiments with MiniZinc")
    
    parser.add_argument("-n", "--instances", type=str, nargs='+', 
                        default=["6", "8", "10", "12", "14", "16", "18"],
                        help="List of instances to run (e.g. 6 8 10)")
    
    parser.add_argument("-t", "--timeout", type=int, default=300,
                        help="Timeout in seconds (default: 300)")
    
    parser.add_argument("--solver", type=str, default="all",
                        help="Specific solver/experiment name to run (default: all)")

    args = parser.parse_args()

    if not os.path.exists(MODEL_DECISION):
        print(f"Error: Model files not found at {BASE_PATH}")
        sys.exit(1)

    print(f"--- Starting CP Experiments ---")
    print(f"Instances: {args.instances}")
    print(f"Timeout: {args.timeout}s")
    print(f"Mode: {args.solver}")
    print("-" * 30)

    for inst in args.instances:
        dzn = os.path.join(DZN_DIR, f"{inst}.dzn")
        json_path = os.path.join(OUTPUT_DIR, f"{inst}.json")
        
        if not os.path.exists(dzn):
            print(f"Skipping {inst}: .dzn file not found.")
            continue
        
        print(f"Processing Instance {inst}...")
        
        data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: data = json.load(f)
            except: pass

        for exp in ALL_EXPERIMENTS:
            if args.solver != "all" and args.solver != exp["name"]:
                continue

            if not os.path.exists(exp["model"]): 
                continue
            
            result = run_solver(exp, dzn, args.timeout)
            data[exp["name"]] = result
            
            status = "OPTIMAL" if result["optimal"] else "TIMEOUT"
            if not result["sol"]: status += "(NO SOL)"
            print(f"  > {exp['name']:<20} : {result['time']}s [{status}]")

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    print("\nAll done.")

if __name__ == "__main__":
    main()
