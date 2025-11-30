include "globals.mzn";

int: n;
constraint n mod 2 = 0;

int: weeks = n - 1;
int: periods = n div 2;
int: N1 = n - 1;

set of int: Teams = 1..n;
set of int: Weeks = 1..weeks;
set of int: Periods = 1..periods;

function int: get_k(int: t, int: w) = 
    (((t - w) mod N1 + N1) mod N1) + 1;

function int: get_team_at_k(int: k, int: w) = 
    ((k + w - 2) mod N1) + 1;

array[Teams, Weeks] of int: opponent = array2d(Teams, Weeks, [
    if t == n then
        get_team_at_k(1, w)
    else
        let { int: k = get_k(t, w); } in
        if k == 1 then n else get_team_at_k(n - k + 1, w) endif
    endif
    | t in Teams, w in Weeks
]);

array[Teams, Weeks] of var Periods: period;
array[Teams, Weeks] of var bool: is_home;


% Each game period is shared by both teams
constraint forall(t in Teams, w in Weeks)(
    period[t,w] = period[opponent[t,w], w]
);

% Each period in a week can host only one match
% Only for teams t < opponent[t,w]
constraint forall(w in Weeks)(
    alldifferent([ period[t,w] | t in Teams where t < opponent[t,w] ])
);

% Each team appears at most twice in the same period over the season
constraint forall(t in Teams)(
    global_cardinality_low_up(
        [ period[t,w] | w in Weeks ],  
        Periods,                        
        [ 0 | _ in Periods ],           
        [ 2 | _ in Periods ]            
    )
);

% Home/Away are opposite
constraint forall(t in Teams, w in Weeks)(
    is_home[t,w] != is_home[opponent[t,w], w]
);

% SYMMETRY BREAKING
constraint is_home[1,1] = true;

constraint period[n, 1] = 1; 
constraint forall(i in 2..periods)( 
    period[i, 1] = i
);


solve :: seq_search([
    int_search(
        [ period[t,w] | w in Weeks, t in Teams where t < opponent[t,w] ],
        first_fail,     
        indomain_max,   
        complete
    ),
    int_search(
        [ is_home[t,w] | w in Weeks, t in Teams where t < opponent[t,w] ],
        first_fail, 
        indomain_max, 
        complete
    )
]) satisfy;

array[Periods, Weeks, 1..2] of var Teams: games;

constraint forall(t in Teams, w in Weeks where t < opponent[t,w])(
    let {
        var Periods: p = period[t,w],
        var Teams: a = if is_home[t,w] then t else opponent[t,w] endif,
        var Teams: b = if is_home[t,w] then opponent[t,w] else t endif
    } in
        games[p, w, 1] = a /\
        games[p, w, 2] = b
);

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
    ]) ++ "\n]\n"
];