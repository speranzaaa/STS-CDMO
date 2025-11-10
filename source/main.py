# main.py

import argparse
import json
import os
import time
import sys
from pathlib import Path

# Add the 'source' directory to the path to allow module imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'source'))

# Import the specific solvers
# Each solver.py file should have a 'solve(n, timeout)' function
from sat.solver import solve_sts_sat

# Mapping of approaches to their solvers and folders
APPROACHES = {
    "sat": {
        "solver": solve_sts_sat,
        "folder": "sat",
        "name": "z3_sat" # Meaningful name for the report 
    },
    # Add other approaches 
}

def main():
    """
    Main function to run the Sports Tournament Scheduling models.
    """
    parser = argparse.ArgumentParser(description="Solver for the Sports Tournament Scheduling (STS) problem.")
    parser.add_argument("n", type=int, help="Number of teams (must be an even number).")
    parser.add_argument("approach", choices=APPROACHES.keys(), help="The modeling approach to use.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for the solver.") # As per project spec 
    
    args = parser.parse_args()

    n = args.n
    approach_key = args.approach
    timeout = args.timeout

    if n % 2 != 0:
        print(f"Error: The number of teams 'n' must be even.") # Based on n/2 periods 
        sys.exit(1)

    print(f"Starting solver for n={n} with approach '{approach_key}' and a {timeout}s timeout.")

    # Select the correct solver
    solver_func = APPROACHES[approach_key]["solver"]
    approach_name = APPROACHES[approach_key]["name"]
    
    start_time = time.time()
    
    # Run the solver
    # The solver function must return: (is_optimal, solution, obj_value)
    # where is_optimal is True if a solution is found (or optimal), False otherwise.
    is_optimal, solution, obj_value = solver_func(n, timeout)
    
    end_time = time.time()
    
    # Calculate runtime as an integer (floor of the actual runtime) 
    runtime = int(end_time - start_time)

    # Handle timeout case as per specification 
    if runtime >= timeout and not is_optimal:
        runtime = timeout
        print(f"Timeout of {timeout} seconds reached.")
    else:
        status_msg = "Solution found" if is_optimal else "No solution found within the time limit"
        print(f"{status_msg} in {runtime} seconds.")

    # Create the results directory if it doesn't exist 
    res_folder = project_root / 'res' / APPROACHES[approach_key]['folder']
    os.makedirs(res_folder, exist_ok=True)
    
    output_path = res_folder / f"{n}.json" # e.g., res/sat/6.json 
    
    # JSON output structure
    output_data = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            output_data = json.load(f)
            
    # Each key in the json is the approach name 
    output_data[approach_name] = {
        "time": runtime,
        "optimal": is_optimal,
        "obj": obj_value,
        "sol": solution
    } # Conforms to the required JSON fields 

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()