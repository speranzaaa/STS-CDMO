from pulp import *
import highspy
def print_tournament_schedule(matches):
    TEAMS = range(1,n+1) # from 1 to n
    WEEKS = range(1,n) # from 1 to n-1
    PERIODS = range (1,(n//2)+1) # from 1 to n//2

    # First, collect all games
    games = []
    for w in WEEKS:
        for p in PERIODS:
            for i in TEAMS:
                for j in TEAMS:
                    if i != j and matches[i][j][w][p].varValue == 1:
                        games.append({
                            'week': w,
                            'period': p,
                            'home': i,
                            'away': j
                        })

    
# Main Schedule Table: Periods (rows) × Weeks (columns)
    print("\n📊 SCHEDULE TABLE (Periods × Weeks)\n")
    # Calculate column width based on number of weeks
    col_width = max(15, 60 // len(WEEKS))  # Adjust column width based on number of weeks

    # Create and print header
    header = "Period".ljust(7) + "│"
    for w in WEEKS:
        header += f" Week {w}".center(col_width) + "│"
    print("┌" + "─" * 7 + "┬" + ("─" * col_width + "┬") * len(WEEKS))
    print("│" + header[:-1])
    print("├" + "─" * 7 + "┼" + ("─" * col_width + "┼") * len(WEEKS))

    # Print each period row
    for p in PERIODS:
        row = f"   {p}   │"

        for w in WEEKS:
            # Find games for this period and week
            period_week_games = [g for g in games if g['period'] == p and g['week'] == w]

            if period_week_games:
                # Format games for this cell
                games_in_cell = []
                for game in period_week_games:
                    games_in_cell.append(f"{game['home']}v{game['away']}")

                cell_content = ", ".join(games_in_cell)

                # Truncate if too long for column
                if len(cell_content) > col_width - 2:
                    cell_content = cell_content[:col_width-5] + "..."

                row += f" {cell_content}".ljust(col_width) + "│"
            else:
                row += " ".ljust(col_width) + "│"

        print("│" + row[:-1])

    # Bottom border
    print("└" + "─" * 7 + "┴" + ("─" * col_width + "┴") * len(WEEKS))

    # Home/Away balance per team
    print(f"\n⚖️  HOME/AWAY BALANCE")
    print("-" * 30)
    print("Team | Home | Away | Diff")
    print("-----|------|------|------")

    for t in TEAMS:
        home_count = len([g for g in games if g['home'] == t])
        away_count = len([g for g in games if g['away'] == t])
        diff_count = abs(home_count - away_count)
        print(f"  {t}  |  {home_count}   |  {away_count}   |   {diff_count}")

    print("="*80)



if __name__ == "__main__":
    n=12
    optimization = True
    games_per_pair = 1
    games_per_team_per_week = 1
    games_per_slot = 1
    max_games_per_team_per_period = 2

    TEAMS = range(1,n+1) # from 1 to n
    WEEKS = range(1,n) # from 1 to n-1
    PERIODS = range (1,(n//2)+1) # from 1 to n//2

    problem = LpProblem("Scheduling_Tournament", LpMinimize)

    #Decision variable
    matches = LpVariable.dicts("Match", (TEAMS, TEAMS, WEEKS, PERIODS), cat="Binary")

    #Constraints

    # 1 every team plays every other team once
    for i in TEAMS:
        for j in TEAMS:
            if i<j:
                problem += lpSum(matches[i][j][w][p] + matches[j][i][w][p] for w in WEEKS for p in PERIODS) == games_per_pair

    # 2 every team plays exactly one game each week
    for t in TEAMS:
        for w in WEEKS:
            problem += lpSum(matches[t][j][w][p] + matches[j][t][w][p] for j in TEAMS if j!=t for p in PERIODS) == games_per_team_per_week

    # 3 every team plays at most 2 times in the same period
    for t in TEAMS:
        for p in PERIODS:
            problem += lpSum(matches[t][j][w][p] + matches[j][t][w][p] for j in TEAMS if j!=t for w in WEEKS) <= max_games_per_team_per_period

    # 4 each slot (week,period) has one game
    for w in WEEKS:
        for p in PERIODS:
            problem += lpSum(matches[i][j][w][p] for i in TEAMS for j in TEAMS if i!=j) == games_per_slot

    # SYMMETRY BREAKING 
    # 1 fix first week
    fixed_matches = []
    for k in range(1,n,2):
        home = k
        away = k+1
        fixed_matches.append((home, away))

    for period, (home_team, away_team) in enumerate(fixed_matches, 1):

        matches[home_team][away_team][2][period].setInitialValue(1)
        matches[home_team][away_team][2][period].fixValue()

        # FIX ALL OTHER MATCHES IN THAT SLOT TO 0
        for i in TEAMS:
            for j in TEAMS:
                if i == j:
                    continue 

                if (i, j) != (home_team, away_team):
                    matches[i][j][2][period].setInitialValue(0)
                    matches[i][j][2][period].fixValue()

    # 2 fix second week
    util_list=[x for x in range(1,n+1) if x%2!=0] + [x for x in range(1,n+1) if x%2==0]
    fixed_matches = []
    for k in range(0,n,2):
        home = util_list[k]
        away = util_list[k+1]
        fixed_matches.append((home, away))

    for period, (home_team, away_team) in enumerate(fixed_matches, 1):

        matches[home_team][away_team][1][period].setInitialValue(1)
        matches[home_team][away_team][1][period].fixValue()

        # FIX ALL OTHER MATCHES IN THAT SLOT TO 0
        for i in TEAMS:
            for j in TEAMS:
                if i == j:
                    continue 

                if (i, j) != (home_team, away_team):
                    matches[i][j][1][period].setInitialValue(0)
                    matches[i][j][1][period].fixValue()

    

    if optimization:
        # 5. Constraints to capture absolute difference between home and away games
        home_away_diff = LpVariable.dicts("HomeAwayDiff", TEAMS, lowBound=1, upBound=n-1, cat="Integer")
        
        for t in TEAMS:
            home_games = lpSum(matches[t][j][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
            away_games = lpSum(matches[j][t][w][p] for j in TEAMS if j!=t for w in WEEKS for p in PERIODS)
            
            problem += home_away_diff[t] >= home_games - away_games
            problem += home_away_diff[t] >= away_games - home_games
        
        #objective: minimize total home/away imbalance across all teams
        problem += lpSum(home_away_diff[t] for t in TEAMS)
    


    #solution
    solver = PULP_CBC_CMD(msg=1, timeLimit=300, warmStart=True) 
    #solver = GLPK_CMD(msg=1, timeLimit=300)
    #solver = HiGHS(msg=1, timeLimit=300, warmStart=True)
    #solver = COIN_CMD(msg=1, timeLimit=300, warmStart=True)
    #solver = SCIP_PY(msg=1, timeLimit=300, warmStart=False) 
    #solver = CYLP(msg=1, timeLimit=300, warmStart=True) 

    problem.solve(solver)
    print("Status:", LpSolution[problem.status])

    print("Time:", problem.solutionCpuTime)
    print("Solution:")
    print_tournament_schedule(matches)


