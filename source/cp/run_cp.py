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
    {
        "name": "gecode",
        "solver": "gecode",
        "model": MODEL_DECISION
    },
    {
        "name": "chuffed",
        "solver": "chuffed",
        "model": MODEL_DECISION
    },
    {
        "name": "gecode_opt",
        "solver": "gecode",
        "model": MODEL_OPT
    }
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_minizinc_output(stdout, is_optimization):
    final_sol = []
    final_obj = None
    is_optimal = False
    
    has_separator = "----------" in stdout
    has_double_separator = "==========" in stdout
    
    if is_optimization:
        is_optimal = has_double_separator
    else:
        is_optimal = has_separator

    if "=====UNSATISFIABLE=====" in stdout:
        return [], None, False

    if has_separator:
        blocks = stdout.split("----------")
        for block in blocks:
            block = block.strip()
            if not block: continue
            try:
                start = block.find('[')
                end = block.rfind(']') + 1
                if start != -1 and end != -1:
                    sol_list = json.loads(block[start:end])
                    
                    obj_match = re.search(r'obj\s*=\s*(\d+)', block)
                    current_obj = int(obj_match.group(1)) if obj_match else None
                    
                    final_sol = sol_list
                    final_obj = current_obj
            except:
                continue
                
    return final_sol, final_obj, is_optimal

def run_experiment(exp_config, dzn_path):
    model_path = exp_config["model"]
    solver_id = exp_config["solver"]
    is_optimization = "opt" in model_path or "opt" in exp_config["name"]

    command = [
        "minizinc",
        "--solver", solver_id,
        "--time-limit", str(TIME_LIMIT_MS),
        model_path,
        dzn_path
    ]

    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC + 10,
            encoding='utf-8'
        )
        stdout = result.stdout.strip()
        actual_runtime = time.perf_counter() - start_time
    except subprocess.TimeoutExpired:
        stdout = ""
        actual_runtime = TIME_LIMIT_SEC
    
    sol, obj, optimal = parse_minizinc_output(stdout, is_optimization)
    
    report_time = math.floor(actual_runtime)
    if report_time >= TIME_LIMIT_SEC:
        report_time = TIME_LIMIT_SEC
        if not sol:
            optimal = False
    
    if not is_optimization and obj is None:
        obj = None

    return {
        "time": report_time,
        "optimal": optimal,
        "obj": obj,
        "sol": sol
    }

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
