from z3 import *

def solve_sts_sat(n: int, timeout: int):
    """
    Solves the STS problem for 'n' teams using a SAT model with Z3.

    Args:
        n (int): The number of teams (must be even).
        timeout (int): Timeout in seconds for the solver.

    Returns:
        tuple: (is_optimal, solution_matrix, objective_value)
               is_optimal (bool): True if a solution was found.
               solution_matrix (list): The schedule matrix, or None.
               objective_value (None): No objective function for the decision version. 
    """
    # Problem parameters based on the description
    weeks = n - 1 # A tournament is scheduled over n-1 weeks 
    periods = n // 2 # Each week is divided into n/2 periods 
    teams = list(range(1, n + 1))

    s = Solver()
    s.set("timeout", timeout * 1000) # Z3 uses milliseconds for timeout

    # --- Decision Variables ---
    # schedule[w][p][s] = team_id
    # w: week, p: period, s: slot (0=home, 1=away) 
    schedule = [[[Int(f"t_w{w}_p{p}_s{s}") for s in range(2)] for p in range(periods)] for w in range(weeks)]

    # Variable domains: each cell must contain a valid team ID
    for w in range(weeks):
        for p in range(periods):
            s.add(And(schedule[w][p][0] >= 1, schedule[w][p][0] <= n))
            s.add(And(schedule[w][p][1] >= 1, schedule[w][p][1] <= n))
            # A team cannot play against itself
            s.add(schedule[w][p][0] != schedule[w][p][1])

    # --- Constraints ---

    # 1. Every team plays once a week. 
    for w in range(weeks):
        for t in teams:
            # Create a list of occurrences of team 't' in week 'w'
            occurrences = []
            for p in range(periods):
                occurrences.append(schedule[w][p][0] == t)
                occurrences.append(schedule[w][p][1] == t)
            # Team 't' must appear exactly once
            s.add(AtMost(*occurrences, 1))
            s.add(AtLeast(*occurrences, 1))
    
    # 2. Every team plays with every other team only once. 
    for t1 in teams:
        for t2 in teams:
            if t1 < t2:
                # matchups will contain a boolean variable for each possible game between t1 and t2
                matchups = []
                for w in range(weeks):
                    for p in range(periods):
                        matchups.append(
                            Or(
                                And(schedule[w][p][0] == t1, schedule[w][p][1] == t2),
                                And(schedule[w][p][0] == t2, schedule[w][p][1] == t1)
                            )
                        )
                # The game between t1 and t2 must occur exactly once
                s.add(AtMost(*matchups, 1))
                s.add(AtLeast(*matchups, 1))

    # 3. Every team plays at most twice in the same period over the tournament. 
    for p in range(periods):
        for t in teams:
            # appearances will contain a boolean variable for each appearance of team 't' in period 'p'
            appearances = []
            for w in range(weeks):
                appearances.append(Or(schedule[w][p][0] == t, schedule[w][p][1] == t))
            # Team 't' can appear at most 2 times in period 'p'
            s.add(AtMost(*appearances, 2))

    # --- Symmetry Breaking (optional but recommended to improve performance) --- 
    # We fix the first game of the first week to break rotational symmetry.
    s.add(schedule[0][0][0] == 1)
    s.add(schedule[0][0][1] == 2)


    # --- Solving and Solution Extraction ---
    if s.check() == sat:
        m = s.model()
        # Format the solution as an (n/2)x(n-1) matrix as required 
        # sol[period][week] = [home, away]
        sol_matrix = [[None for _ in range(weeks)] for _ in range(periods)]
        for w in range(weeks):
            for p in range(periods):
                home_team = m.eval(schedule[w][p][0]).as_long()
                away_team = m.eval(schedule[w][p][1]).as_long()
                sol_matrix[p][w] = [home_team, away_team]
        
        return True, sol_matrix, None # optimal=True, sol=matrix, obj=None 
    else:
        return False, None, None # optimal=False, sol=None, obj=None