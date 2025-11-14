import subprocess
import json
import os
import math
import time

MODEL_FILE = os.path.join("source", "CP", "cp_model.mzn")
DZN_DIR = os.path.join("source", "CP")
INSTANCES = ["6", "8", "10", "12", "14", "16"] 
SOLVER = "Gecode"
APPROACH_NAME = "Gecode"
TIME_LIMIT_MS = 300000
TIME_LIMIT_SEC = 300

output_dir = os.path.join("res", "CP")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    exit()

for inst_name in INSTANCES:
    dzn_file = os.path.join(DZN_DIR, f"{inst_name}.dzn")
    json_file_path = os.path.join(output_dir, f"{inst_name}.json")

    if not os.path.exists(dzn_file):
        print(f"Warning: Data file not found at {dzn_file}. Skipping instance.")
        continue

    command = [
        "minizinc",
        "--solver", SOLVER,
        "--time-limit", str(TIME_LIMIT_MS),
        MODEL_FILE,
        dzn_file
    ]

    start_time = time.perf_counter()
    sol_data = []
    optimal = False
    runtime_sec = 0

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC + 1,
            encoding='utf-8'
        )
        
        end_time = time.perf_counter()
        runtime_sec = end_time - start_time
        stdout = result.stdout.strip()
        stderr = result.stderr

        if "=====UNSATISFIABLE=====" in stdout:
            optimal = False
            sol_data = []
        elif stdout:
            try:
                solution_str = stdout.split("----------")[0].strip()
                sol_data = json.loads(solution_str)
                optimal = True
            except json.JSONDecodeError:
                optimal = False
                sol_data = [] 
                runtime_sec = TIME_LIMIT_SEC 
                print(f"--- ERROR: Instance {inst_name} output was not valid JSON! ---")
                print(stdout)
                print(f"--------------------------------------------------")
        else:
            optimal = False
            sol_data = [] 
            if runtime_sec < TIME_LIMIT_SEC:
                runtime_sec = TIME_LIMIT_SEC
            if stderr:
                print(f"--- ERROR: Instance {inst_name} produced an error ---")
                print(stderr)
                print(f"---------------------------------------")

    except subprocess.TimeoutExpired:
        runtime_sec = TIME_LIMIT_SEC
        optimal = False
        sol_data = []
    
    if runtime_sec >= TIME_LIMIT_SEC and not optimal:
        run_time_int = TIME_LIMIT_SEC
    else:
        run_time_int = math.floor(runtime_sec)

    approach_result = {
        "time": run_time_int,
        "optimal": optimal,
        "obj": None, 
        "sol": sol_data
    }

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    all_results[APPROACH_NAME] = approach_result

    with open(json_file_path, 'w') as f:
        json.dump(all_results, f)