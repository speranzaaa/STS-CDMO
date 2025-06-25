% Sports Tournament Scheduling - CP

int: n;
int: weeks = n - 1;
int: periods = n div 2;
set of int: Teams = 1..n;
array[1..weeks, 1..periods, 1..2] of var Teams: games;
% games[w, p, 1] is the home team in week w, period p；games[w, p, 2] is the away team

% home team ≠ away team
constraint
  forall(w in 1..weeks, p in 1..periods) (
    games[w, p, 1] != games[w, p, 2]
  );

% Check if each team plays once per week
constraint
  forall(w in 1..weeks, t in Teams) (
    sum([
      bool2int(games[w, p, 1] == t \/ games[w, p, 2] == t)
      | p in 1..periods
    ]) == 1
  );

% Check if each match occurs exactly once
constraint
  forall(i in Teams, j in Teams where i < j) (
    sum([
      bool2int(
        (games[w, p, 1] == i /\ games[w, p, 2] == j) \/
        (games[w, p, 1] == j /\ games[w, p, 2] == i)
      )
      | w in 1..weeks, p in 1..periods
    ]) == 1
  );

% A team appears in same period at most 2 times
constraint
  forall(p in 1..periods, t in Teams) (
    sum([
      bool2int(games[w, p, 1] == t \/ games[w, p, 2] == t)
      | w in 1..weeks
    ]) <= 2
  );

% Symmetry breaking : Team 1 plays first game at home
constraint
  games[1, 1, 1] == 1;

solve satisfy;

% Output formatted
output [
  "[\n" ++
  concat([
    "  [" ++
    concat([
      "[" ++ show(games[w, p, 1]) ++ "," ++ show(games[w, p, 2]) ++ "]"
      ++ (if p < periods then ", " else "" endif)
      | p in 1..periods
    ]) ++ "]"
    ++ (if w < weeks then ",\n" else "\n" endif)
    | w in 1..weeks
  ]) ++ "]\n"
];
