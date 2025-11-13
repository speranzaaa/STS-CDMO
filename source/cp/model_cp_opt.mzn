include "globals.mzn";

int: n;
constraint n mod 2 = 0;

int: weeks = n - 1;
int: periods = n div 2;
set of int: Teams = 1..n;
set of int: Weeks = 1..weeks;
set of int: Periods = 1..periods;

array[Teams, Weeks] of var Teams: opponent;
array[Teams, Weeks] of var Periods: period;
array[Teams, Weeks] of var bool: is_home;
array[Periods, Weeks, 1..2] of var Teams: games;

% CORE CONSTRAINTS (Round-Robin Logic)
constraint forall(w in Weeks)(
    let {
        array[1..(n - 1)] of int: teams = [((i + w - 2) mod (n - 1)) + 1 | i in 1..(n - 1)]
    } in
        (
            % central team n plays against team in position 1
            opponent[teams[1], w] = n /\
            opponent[n, w] = teams[1] /\
            
            % the remaining teams pair up symmetrically
            forall(i in 2..(n div 2)) (
                let {
                    int: t1 = teams[i],
                    int: t2 = teams[n - i + 1]
                } in
                    opponent[t1, w] = t2 /\
                    opponent[t2, w] = t1
            )
        )
);

% Each game period is shared by both teams
constraint forall(t in Teams, w in Weeks)(
    period[t,w] = period[opponent[t,w], w]
);

% Each period in a week can host only one match
constraint forall(w in Weeks)(
    alldifferent([ period[t,w] | t in Teams where t < opponent[t,w] ])
);

% Each team appears at most twice in the same period over the season
constraint forall(t in Teams, p in Periods)(
    sum([ bool2int(period[t,w] == p) | w in Weeks ]) <= 2
);

% Home/Away are opposite
constraint forall(t in Teams, w in Weeks)(
    is_home[t,w] != is_home[opponent[t,w], w]
);

% Build games matrix (home/away pairing)
constraint forall(t in Teams, w in Weeks)(
    if t < opponent[t,w] then
        let {
            var Periods: p = period[t,w],
            var Teams: a = if is_home[t,w] then t else opponent[t,w] endif,
            var Teams: b = if is_home[t,w] then opponent[t,w] else t endif
        } in
            games[p, w, 1] = a /\
            games[p, w, 2] = b
    else
        true
    endif
);

% Each team must appear once per week (Redundant)
constraint forall(w in Weeks)(
    alldifferent([ games[p, w, s] | p in Periods, s in 1..2 ])
);

% Each team plays at most twice in the same period (Redundant)
constraint forall(p in Periods)(
    global_cardinality_low_up(
        [ games[p, w, s] | w in Weeks, s in 1..2 ],
        [ t | t in Teams ], [0 | t in Teams], [2 | t in Teams]
    )
);

% SYMMETRY BREAKING
% Break home/away symmetry for one game
constraint is_home[1,1] = true;

constraint period[n, 1] = 1;
constraint forall(i in 2..periods)(
    period[i, 1] = i
);


% OPTIMIZATION: BALANCE HOME/AWAY
array[Teams] of var 0..weeks: home_count = [
    sum([ bool2int(is_home[t,w]) | w in Weeks ]) | t in Teams
];
% imbalance = 1
% abs(2 * home_count[t] - weeks)
constraint forall(t in Teams)(
    abs(2 * home_count[t] - weeks) = 1
);

solve :: seq_search([
    int_search([ period[t,w] | t in Teams, w in Weeks ], first_fail, indomain_min, complete),
    int_search([ is_home[t,w] | t in Teams, w in Weeks ], first_fail, indomain_min, complete)
]) satisfy;

output [
    "[\n" ++
    concat([
        "  [" ++
        concat([
            "[" ++ show(games[p, w, 1]) ++ "," ++ show(games[p, w, 2]) ++ "]"
            ++ if w < weeks then ", " else "" endif
            | w in Weeks
        ]) ++ "]"
        ++ if p < periods then ",\n" else "\n" endif
        | p in Periods
    ]) ++ "\n]\n",
];