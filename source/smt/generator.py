def generate_smtfile_QFUF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even")

    with open(filename, "w") as f:
        f.write("(set-logic QF_UFLIA)\n")  # Set the theory
        f.write("(set-option :produce-models true)\n")

        # Variables Definition
        f.write("(declare-fun home (Int Int) Int)\n")  # home(week, period)
        f.write("(declare-fun away (Int Int) Int)\n")  # away(week, period)

        # Variables Domain Definition + Constraint
        f.write("(assert (and")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f" (>= (home {w} {p}) 1) (<= (home {w} {p}) {n}) (>= (away {w} {p}) 1) (<= (away {w} {p}) {n}) "
                        f"(not (= (home {w} {p}) (away {w} {p})))\n")
        f.write("))\n")
        # a team can not play against itself

        # Constraint 1: every team plays with every other team only once
        for t1 in range(1, n+1):
            for t2 in range(t1+1, n+1):
                games = []  # games t1 vs. t2 or t2 vs. t1, for each pair
                for w in range(1, n):
                    for p in range(1, (n//2)+1):
                        games.append(f"(ite (and (= (home {w} {p}) {t1}) (= (away {w} {p}) {t2})) 1 0)")
                        games.append(f"(ite (and (= (home {w} {p}) {t2}) (= (away {w} {p}) {t1})) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 2: every team plays once a week
        for w in range(1, n):
            for t in range(1, n+1):
                games = []  # games in a week w in which t plays
                for p in range(1, (n//2)+1):
                    games.append(f"(ite (= (home {w} {p}) {t}) 1 0)")
                    games.append(f"(ite (= (away {w} {p}) {t}) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 3: every team plays at most twice in the same period over the tournament
        for t in range(1, n+1):
            for p in range(1, (n//2)+1):
                games = []  # games in a period p in which t plays
                for w in range(1, n):
                    games.append(f"(ite (= (home {w} {p}) {t}) 1 0)")
                    games.append(f"(ite (= (away {w} {p}) {t}) 1 0)\n")
                f.write(f"(assert (<= (+ {' '.join(games)}\t) 2))\n")

        f.write("(check-sat)\n")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f"(get-value ((home {w} {p}) (away {w} {p})))\n")
        f.close()


def generate_smtfile_QF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even")

    with open(filename, "w") as f:
        f.write("(set-logic QF_LIA)\n")  # Set the theory
        f.write("(set-option :produce-models true)\n")

        # Variables Definition
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f"(declare-const home_{w}_{p} Int)\n")  # home(week, period)
                f.write(f"(declare-const away_{w}_{p} Int)\n")  # away(week, period)

        # Variables Domain Definition + Constraint
        f.write("(assert (and")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f" (>= home_{w}_{p} 1) (<= home_{w}_{p} {n}) (>= away_{w}_{p} 1) (<= away_{w}_{p} {n}) "
                        f"(not (= home_{w}_{p} away_{w}_{p}))\n")
        f.write("))\n")
        # a team can not play against itself

        # Constraint 1: every team plays with every other team only once
        for t1 in range(1, n+1):
            for t2 in range(t1+1, n+1):
                games = []  # games t1 vs. t2 or t2 vs. t1, for each pair
                for w in range(1, n):
                    for p in range(1, (n//2)+1):
                        games.append(f"(ite (and (= home_{w}_{p} {t1}) (= away_{w}_{p} {t2})) 1 0)")
                        games.append(f"(ite (and (= home_{w}_{p} {t2}) (= away_{w}_{p} {t1})) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 2: every team plays once a week
        for w in range(1, n):
            for t in range(1, n+1):
                games = []  # games in a week w in which t plays
                for p in range(1, (n//2)+1):
                    games.append(f"(ite (= home_{w}_{p} {t}) 1 0)")
                    games.append(f"(ite (= away_{w}_{p} {t}) 1 0)\n")
                f.write(f"(assert (= 1 (+ {' '.join(games)}\t)))\n")

        # Constraint 3: every team plays at most twice in the same period over the tournament
        for t in range(1, n+1):
            for p in range(1, (n//2)+1):
                games = []  # games in a period p in which t plays
                for w in range(1, n):
                    games.append(f"(ite (= home_{w}_{p} {t}) 1 0)")
                    games.append(f"(ite (= away_{w}_{p} {t}) 1 0)\n")
                f.write(f"(assert (<= (+ {' '.join(games)}\t) 2))\n")

        f.write("(check-sat)\n")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f"(get-value (home_{w}_{p} away_{w}_{p}))\n")
        f.close()


def generate_smtfile_UF(n, filename="schedule.smt2"):
    # Check if the number of teams is even
    if n % 2 != 0:
        raise ValueError("n must be even")

    with open(filename, "w") as f:
        f.write("(set-logic UFLIA)\n")  # Set the theory

        # Variables Definition
        f.write("(declare-fun home (Int Int) Int)\n")  # home(week, period)
        f.write("(declare-fun away (Int Int) Int)\n")  # away(week, period)

        # Variables Domain Definition + Constraint
        f.write("(assert (forall ((w Int) (p Int))\n")
        f.write(f"\t(and (>= (home w p) 1) (<= (home w p) {n})\n")  # 1 <= home(w,p) <= n
        f.write(f"\t\t(>= (away w p) 1) (<= (away w p) {n})\n")  # 1 <= away(w,p) <= n
        f.write(f"\t\t(not (= (home w p) (away w p))))))\n")  # home(w,p) not equal to away(w,p)
        # a team can not play against itself

        # Constraint 1: every team plays with every other team only once
        f.write("(assert (forall ((i Int) (j Int))"
                f"(=> (and (>= i 1) (<= i {n}) (>= j 1) (<= j {n}) (not (= i j)))\n"
                "(=  1 (+ ")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f"(ite (and (= (home {w} {p}) i) (= (away {w} {p}) j)) 1 0)\n"
                        f"(ite (and (= (home {w} {p}) j) (= (away {w} {p}) i)) 1 0)\n")
        f.write(")))))\n")

        # Constraint 2: every team plays once a week
        f.write("(assert (forall ((w Int) (t Int))"
                f"(=> (and (>= w 1) (<= w {n-1}) (>= t 1) (<= t {n}))"
                "(exists ((p Int))"
                f"(and (>= p 0) (< p {n//2})"
                "(or (= (home w p) t) (= (away w p) t)))))))\n")

        # Constraint 3: every team plays at most twice in the same period over the tournament
        f.write("(assert (forall ((t Int) (p Int))"
                f"(=> (and (>= t 1) (<= t {n}) (>= p 1) (<= p {n//2}))\n"
                f"(<= 2 (+ ")
        for w in range(1, n):
            f.write(f"(ite (or (= (home {w} p) t) (= (away {w} p) t)) 1 0)\n")
        f.write(")))))\n")

        f.write("(check-sat)\n")
        for w in range(1, n):
            for p in range(1, (n//2)+1):
                f.write(f"(get-value ((home {w} {p}) (away {w} {p})))\n")
        f.close()


generate_smtfile_QF(8)

