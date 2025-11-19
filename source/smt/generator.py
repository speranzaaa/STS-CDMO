import subprocess
import time
from pysmt.shortcuts import Solver, Symbol, Not, Plus, Ite, Int, Equals
from pysmt.typing import BOOL
from pysmt.smtlib.parser import SmtLibParser
import json


# Model without the use of the circle method (kirkman algorithm) - QF_UFLIA logic
def generate_smtfile_QFUF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even!")

    with open(filename, "w") as f:
        f.write("(set-option :produce-models true)\n")
        f.write("(set-logic QF_UFLIA)\n")  # Set the theory

        # Variables Definition
        f.write("(declare-fun home (Int Int) Int)\n")  # home(week, period)
        f.write("(declare-fun away (Int Int) Int)\n")  # away(week, period)

        # Variables Domain Definition + Constraint
        f.write("(assert (and")
        for w in range(1, n):
            for p in range(1, (n // 2) + 1):
                f.write(f" (>= (home {w} {p}) 1) (<= (home {w} {p}) {n}) (>= (away {w} {p}) 1) (<= (away {w} {p}) {n}) "
                        f"(not (= (home {w} {p}) (away {w} {p})))\n")
        f.write("))\n")
        # a team can not play against itself

        # Constraint 1: every team plays with every other team only once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                games = []  # games t1 vs. t2 or t2 vs. t1, for each pair
                for w in range(1, n):
                    for p in range(1, (n // 2) + 1):
                        games.append(f"(ite (and (= (home {w} {p}) {t1}) (= (away {w} {p}) {t2})) 1 0)")
                        games.append(f"(ite (and (= (home {w} {p}) {t2}) (= (away {w} {p}) {t1})) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 2: every team plays once a week
        for w in range(1, n):
            for t in range(1, n + 1):
                games = []  # games in a week w in which t plays
                for p in range(1, (n // 2) + 1):
                    games.append(f"(ite (= (home {w} {p}) {t}) 1 0)")
                    games.append(f"(ite (= (away {w} {p}) {t}) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 3: every team plays at most twice in the same period over the tournament
        for t in range(1, n + 1):
            for p in range(1, (n // 2) + 1):
                games = []  # games in a period p in which t plays
                for w in range(1, n):
                    games.append(f"(ite (= (home {w} {p}) {t}) 1 0)")
                    games.append(f"(ite (= (away {w} {p}) {t}) 1 0)\n")
                f.write(f"(assert (<= (+ {' '.join(games)}\t) 2))\n")

        f.write("(check-sat)\n")
        for w in range(1, n):
            for p in range(1, (n // 2) + 1):
                f.write(f"(get-value ((home {w} {p}) (away {w} {p})))\n")
        f.close()


# Model without the use of the circle method (kirkman algorithm) - QF_LIA logic
def generate_smtfile_QF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even")

    with open(filename, "w") as f:
        f.write("(set-option :produce-models true)\n")
        f.write("(set-logic QF_LIA)\n")  # Set the theory

        # Variables Definition
        for w in range(1, n):
            for p in range(1, (n // 2) + 1):
                f.write(f"(declare-const home_{w}_{p} Int)\n")  # home(week, period)
                f.write(f"(declare-const away_{w}_{p} Int)\n")  # away(week, period)

        # Variables Domain Definition + Constraint
        f.write("(assert (and")
        for w in range(1, n):
            for p in range(1, (n // 2) + 1):
                f.write(f" (>= home_{w}_{p} 1) (<= home_{w}_{p} {n}) (>= away_{w}_{p} 1) (<= away_{w}_{p} {n}) "
                        f"(not (= home_{w}_{p} away_{w}_{p}))\n")
        f.write("))\n")
        # a team can not play against itself

        # Constraint 1: every team plays with every other team only once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                games = []  # games t1 vs. t2 or t2 vs. t1, for each pair
                for w in range(1, n):
                    for p in range(1, (n // 2) + 1):
                        games.append(f"(ite (and (= home_{w}_{p} {t1}) (= away_{w}_{p} {t2})) 1 0)")
                        games.append(f"(ite (and (= home_{w}_{p} {t2}) (= away_{w}_{p} {t1})) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 2: every team plays once a week
        for w in range(1, n):
            for t in range(1, n + 1):
                games = []  # games in a week w in which t plays
                for p in range(1, (n // 2) + 1):
                    games.append(f"(ite (= home_{w}_{p} {t}) 1 0)")
                    games.append(f"(ite (= away_{w}_{p} {t}) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 3: every team plays at most twice in the same period over the tournament
        for t in range(1, n + 1):
            for p in range(1, (n // 2) + 1):
                games = []  # games in a period p in which t plays
                for w in range(1, n):
                    games.append(f"(ite (= home_{w}_{p} {t}) 1 0)")
                    games.append(f"(ite (= away_{w}_{p} {t}) 1 0)\n")
                f.write(f"(assert (<= (+ {' '.join(games)}\t) 2))\n")

        f.write("(check-sat)\n")
        for w in range(1, n):
            for p in range(1, (n // 2) + 1):
                f.write(f"(get-value (home_{w}_{p} away_{w}_{p}))\n")
        f.close()


# Model exploiting Kirkman's algorithm without symmetry breaking  - QF_LIA logic
def generate_smtfile_CM(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even!")

    weeks = n - 1
    periods = n // 2
    team_matches = dict((i, list()) for i in range(1, n + 1))
    # a dictionary where the keys are the teams and the values are the matches of that team

    schedule = kirkman_tournament(n)
    with open(filename, "w") as f:
        f.write("(set-option :produce-models true)\n")
        f.write("(set-logic QF_LIA)\n")  # Set the theory

        # Variables Definition
        for week in schedule:
            for (t1, t2) in week:
                team_matches[t1].append((t1, t2))
                team_matches[t2].append((t1, t2))
                for i in range(1, periods + 1):
                    f.write(f"(declare-const m_{t1}_{t2}_p{i} Bool)\n")

        # Additional Constraint N. 1: Each match assigned to exactly one period
        for week in schedule:
            for (t1, t2) in week:
                f.write("(assert (= 1 (+")
                for i in range(1, periods + 1):
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Additional Constraint N. 2: Each period per week holds exactly one match
        for week in schedule:
            for i in range(1, periods + 1):
                f.write("(assert (= 1 (+")
                for (t1, t2) in week:
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Constraint N. 3: Every team plays at most twice in the same period
        for team, matches in team_matches.items():
            for i in range(1, periods + 1):
                f.write("(assert (<= (+")
                for (t1, t2) in matches:
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(") 2))\n")

        f.write("(check-sat)\n")
        for week in schedule:
            for (t1, t2) in week:
                for p in range(1, periods + 1):
                    f.write(f"(get-value (m_{t1}_{t2}_p{p}))\n")
        f.close()


# Model exploiting Kirkman's algorithm with symmetry breaking (1) - QF_UFLIA logic
def generate_smtfile_CM_UF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even!")

    weeks = n - 1
    periods = n // 2
    team_matches = dict((i, list()) for i in range(1, n + 1))
    # a dictionary where the keys are the teams and the values are the matches of that team

    schedule = kirkman_tournament(n)
    # print(schedule)
    with open(filename, "w") as f:
        f.write("(set-option :produce-models true)\n")
        f.write("(set-logic QF_UFLIA)\n")  # Set the theory

        # Variables Definition
        p = 1
        for (t1, t2) in schedule[0]:
            team_matches[t1].append((t1, t2))
            team_matches[t2].append((t1, t2))
            f.write(f"(declare-fun m_{t1}_{t2}_p{p} () Bool)\n")
            p = p + 1

        for week in schedule[1:]:
            for (t1, t2) in week:
                team_matches[t1].append((t1, t2))
                team_matches[t2].append((t1, t2))
                for i in range(1, periods + 1):
                    f.write(f"(declare-fun m_{t1}_{t2}_p{i} () Bool)\n")

        # Fixing the first week (symmetry breaking)
        p = 1
        for (t1, t2) in schedule[0]:
            f.write(f"(assert m_{t1}_{t2}_p{p})\n")
            p = p + 1

        # Additional Constraint N. 1: Each match assigned to exactly one period
        for week in schedule[1:]:
            for (t1, t2) in week:
                f.write("(assert (= 1 (+")
                for i in range(1, periods + 1):
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Additional Constraint N. 2: Each period per week holds exactly one match
        for week in schedule[1:]:
            for i in range(1, periods + 1):
                f.write("(assert (= 1 (+")
                for (t1, t2) in week:
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Constraint N. 3: Every team plays at most twice in the same period
        p = 1
        for (t1, t2) in schedule[0]:
            matches_t1 = team_matches[t1]
            for i in range(1, periods + 1):
                f.write("(assert (<= (+")
                for (x1, x2) in matches_t1:
                    if (x1, x2) == (t1, t2) and p == i:
                        f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                    elif (x1, x2) != (t1, t2):
                        f.write(f" (ite m_{x1}_{x2}_p{i} 1 0)")
                f.write(") 2))\n")

            matches_t2 = team_matches[t2]
            for i in range(1, periods + 1):
                f.write("(assert (<= (+")
                for (x1, x2) in matches_t2:
                    if (x1, x2) == (t1, t2) and p == i:
                        f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                    elif (x1, x2) != (t1, t2):
                        f.write(f" (ite m_{x1}_{x2}_p{i} 1 0)")
                f.write(") 2))\n")
            p = p + 1

        f.write("(check-sat)\n")
        p = 1
        for (t1, t2) in schedule[0]:
            f.write(f"(get-value (m_{t1}_{t2}_p{p}))\n")
            p = p + 1

        for week in schedule[1:]:
            for (t1, t2) in week:
                for p in range(1, periods + 1):
                    f.write(f"(get-value (m_{t1}_{t2}_p{p}))\n")
        f.close()


# Models exploiting Kirkman's algorithm with symmetry breaking (2) - QF_UFLIA logic
def generate_smtfile_CM_UF_SB(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even!")

    weeks = n - 1
    periods = n // 2
    team_matches = dict((i, list()) for i in range(1, n + 1))
    # a dictionary where the keys are the teams and the values are the matches of that team

    schedule = kirkman_tournament(n)

    with open(filename, "w") as f:
        f.write("(set-option :produce-models true)\n")
        f.write("(set-logic QF_UFLIA)\n")  # Set the theory

        # Variables Definition
        for week in schedule:
            for (t1, t2) in week:
                team_matches[t1].append((t1, t2))
                team_matches[t2].append((t1, t2))
                for i in range(1, periods + 1):
                    f.write(f"(declare-fun m_{t1}_{t2}_p{i} () Bool)\n")

        # Fix First Week (Symmetry Breaking)
        p = 1
        for (t1, t2) in schedule[0]:
            for i in range(1, periods + 1):
                if i == p:
                    f.write(f"(assert m_{t1}_{t2}_p{i})\n")
                else:
                    f.write(f"(assert (not m_{t1}_{t2}_p{i}))\n")
            p = p + 1

        # Additional Constraint N. 1: Each match assigned to exactly one period
        for week in schedule:
            for (t1, t2) in week:
                f.write("(assert (= 1 (+")
                for i in range(1, periods + 1):
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Additional Constraint N. 2: Each period per week holds exactly one match
        for week in schedule:
            for i in range(1, periods + 1):
                f.write("(assert (= 1 (+")
                for (t1, t2) in week:
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(")))\n")
        f.write("\n")

        # Constraint N. 3: Every team plays at most twice in the same period
        for team, matches in team_matches.items():
            for i in range(1, periods + 1):
                f.write("(assert (<= (+")
                for (t1, t2) in matches:
                    f.write(f" (ite m_{t1}_{t2}_p{i} 1 0)")
                f.write(") 2))\n")

        f.write("(check-sat)\n")
        for week in schedule:
            for (t1, t2) in week:
                for p in range(1, periods + 1):
                    f.write(f"(get-value (m_{t1}_{t2}_p{p}))\n")
        f.close()


# too much time
def model(n, solver_name: str):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even!")

    weeks = n - 1
    periods = n // 2
    team_matches = dict((i, list()) for i in range(1, n + 1))

    schedule = kirkman_tournament(n)

    # Variables Definition
    variables = dict()  # (team1, team2, period) -> Symbol
    for week in schedule:
        for (t1, t2) in week:
            team_matches[t1].append((t1, t2))
            team_matches[t2].append((t1, t2))
            for i in range(1, periods + 1):
                variables[(t1, t2, i)] = Symbol(f"m_{t1}_{t2}_p{i}", BOOL)

    with Solver(name=solver_name, logic="QF_UFLIA") as solver:

        # Fix First Week (Symmetry Breaking)
        p = 1
        for (t1, t2) in schedule[0]:
            for i in range(1, periods + 1):
                if i == p:
                    solver.add_assertion(variables[(t1, t2, i)])
                else:
                    solver.add_assertion(Not(variables[(t1, t2, i)]))
            p = p + 1

        # Additional Constraint N. 1: Each match assigned to exactly one period
        for week in schedule:
            for (t1, t2) in week:
                sum_variables = Plus([Ite(variables[(t1, t2, i)], Int(1), Int(0)) for i in range(1, periods + 1)])
                solver.add_assertion(Equals(sum_variables, Int(1)))

        # Additional Constraint N. 2: Each period per week holds exactly one match
        for week in schedule:
            for i in range(1, periods + 1):
                sum_variables = Plus([Ite(variables[(t1, t2, i)], Int(1), Int(0)) for (t1, t2) in week])
                solver.add_assertion(Equals(sum_variables, Int(1)))

        # Constraint N. 3: Every team plays at most twice in the same period
        for team, matches in team_matches.items():
            for i in range(1, periods + 1):
                sum_variables = Plus([Ite(variables[(t1, t2, i)], Int(1), Int(0)) for (t1, t2) in matches])
                solver.add_assertion(sum_variables <= Int(2))

        # solve the model
        if solver.solve():
            print("SAT")
        else:
            print("UNSAT")  # to


# too much time
def solve_from_file(file_path, solver_name):
    parser = SmtLibParser()
    script = parser.get_script_fname(file_path)  # parsing the .smt2 file

    with Solver(name=solver_name, logic="QF_UFLIA") as solver:

        script.evaluate(solver)  # loading the assertions

        if solver.solve():
            print("SAT")
        else:
            print("UNSAT")
        # result = solver.solve()
        # print("SAT" if result else "UNSAT")
        return solver


# Circle method
def kirkman_tournament(n):
    """
    Parameters
    ----------
    n : int
     the number of teams that participate in the tournament

    Returns
    --------
    list[list[tuple]]
        a tournament defined by the Kirkman algorithm
    """
    tournament = []

    # Define week 1
    week1 = list()
    for i in range(n // 2):
        week1.append((n - i, 1 + i))

    tournament.append(week1)

    # Construct the remaining weeks adding i to each round 1 + i
    for i in range(1, n - 1):
        if i % 2 == 0:
            week = [(n, 1 + i)]
        else:
            week = [(1 + i, n)]
        for j in range(1, n // 2):
            h = week1[j][0] + i
            if h > (n - 1):
                h = h - (n - 1)

            a = week1[j][1] + i
            if a > n - 1:
                a = a - (n - 1)

            week.append((h, a))
        tournament.append(week)

    return tournament

n = 14
s = time.time()
generate_smtfile_CM_UF_SB(n)
solution = subprocess.getoutput("z3 schedule.smt2")
e = time.time()
runtime = e - s
print(runtime)

solution = solution.split("\n")  # splitting the file
if solution[0] == "sat":  # checking if the solution is sat
    schedule = []  # list for the periods
    for _ in range(n//2):
        schedule.append([])

    for s in solution[1:]:  # getting the matches from the solution
        s = s.split(" ")
        if s[1][0] == "t":
            match = s[0].split("_")
            period = int(match[-1][1:])
            schedule[period-1].append([int(match[1]), int(match[2])])

    for s in schedule:
        print(s)

    approach = {
        "time": runtime,
        "optimal": False,
        "obj": None,
        "sol": schedule
    }

    print(json.dumps(approach))

