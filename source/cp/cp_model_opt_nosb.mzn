include "globals.mzn";

int: n;
constraint n mod 2 = 0;

int: weeks = n - 1;
int: periods = n div 2;
int: N1 = n - 1; 

set of int: Teams = 1..n;
set of int: Weeks = 1..weeks;
set of int: Periods = 1..periods;

% Canonical Schedule
function int: team_at_pos(int: pos, int: w) = 
    ((pos + w - 2) mod N1) + 1;

function int: get_opponent_for_team(int: t, int: w) = 
    let {
        int: t_pos = if t == n then -1 
                     else min([ i | i in 1..N1 where team_at_pos(i, w) == t]) 
                     endif;
    } in
    if t_pos == -1 then 
        team_at_pos(1, w)
    elseif t_pos == 1 then
        n
    else
        team_at_pos(n - t_pos + 1, w)
    endif;

array[Teams, Weeks] of int: opponent = array2d(Teams, Weeks, [
    get_opponent_for_team(t, w) | t in Teams, w in Weeks
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

array[Teams] of var int: home_count = [
    sum(w in Weeks)(bool2int(is_home[t,w])) | t in Teams
];


constraint forall(t in Teams)(
    abs(2 * home_count[t] - weeks) = 1
);

% % SYMMETRY BREAKING
% constraint is_home[1,1] = true;

% constraint period[n, 1] = 1; 
% constraint forall(i in 2..periods)( 
%     period[i, 1] = i
% );

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