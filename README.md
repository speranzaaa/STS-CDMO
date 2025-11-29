# CDMO Project – Sports Tournament Scheduling (STS)

This repository contains the source code, experimental results, and report for the **Combinatorial Decision Making and Optimization (CDMO)** project work, academic year 2024/2025.

##  Project Description

The goal of the project is to model and solve the **Sports Tournament Scheduling (STS)** problem, where `n` teams compete over `n-1` weeks. Each team must:
- Play every other team exactly once.
- Play once per week.
- Play in the same period at most twice in the tournament.

The problem was tackled using three combinatorial optimization paradigms:
- Constraint Programming (CP)
- Satisfiability/Satisfiability Modulo Theories (SAT/SMT)
- Mixed-Integer Linear Programming (MILP)

An optional objective was also considered: minimizing the unbalance between home and away games for each team.

##  INFO

All models can be run through Docker to ensure reproducibility.


## HOW TO BUILD & RUN
### 1) Build docker container
```
docker build -t <image-name> .
```

### 2) Run bash inside container
```
docker run -it -v ${PWD}:/app <image-name> bash
```

### 3) Running instructions
Note that all possible params are specified in params.txt
Make sure to set LF as EOF sequence

* Run a single model 
```
# CP
python3 source/CP/run_cp.py -n n -t time --solver solver

# SAT
python3 source/SAT/run_sat.py \
    -n <NUM_TEAMS_OR_RANGE> \
    --exactly_one_encoding <np|bw|seq|heule> \
    --at_most_k_encoding <np|seq|totalizer> \
    --run_decisional | --run_optimization \
    [--sb | --no_sb] \
    [--timeout <SECONDS>] \
    [--verbose] \
    [--save_json]

# SMT
python3 source/SMT/smt_model.py -n n --solver solver --mode mode [-nosb]

# MIP
python3 source/MIP/run_mip.py --mode single --n n --time_limit time --solver_type solver --opt opt --sym sym
```

* Run all model instances fixing n
```
./run_all_on_n.sh n
```

* Run all model instances on all n
```
./run_all.sh
```

### 3) Check instructions
choose model between CP, SAT, SMT, MIP
```
python3 solution_checker.py ./res/<model>
```


# Parameters

## CP
- solver: "gecode", "chuffed", "gecode_opt", "gecode_without_sb", "chuffed_without_sb", "gecode_opt_without_sb" 

## SAT
- exactly_one_encoding: "np", "bw", "seq", "heule"
- at_most_k_encoding: "np", "seq", "totalizer"
- sym: "sb", "no_sb"
## SMT
- solver: "z3", "optimathsat", "all"
- mode: "decisional", "optimal", "all"
- [-nosb]
- [-all]
## MIP
- solver_type: "cbc", "highs", "sciPy"
- opt: "homeAwayDiff0", "homeAwayDiff1", "feasible"
- sym: "circleMethod", "fixWeek1", "fixWeek2", "noSym"
