### https://en.wikipedia.org/wiki/Stanford_Research_Institute_Problem_Solver

prompt_question="""
Task: Solve this STRIPS problem.
"""

lst_tasks = [
    """
    Initial state: At(A), Level(low), BoxAt(C), BananasAt(B)
    Goal state:    Have(bananas)
    """,
    
    """
    Initial state: At(D), Level(low), BoxAt(A), BananasAt(C)
    Goal state: Have(bananas)
    """,
    
    """
    Initial state: At(B), Level(low), BoxAt(D), BananasAt(A)
    Goal state: Have(bananas)
    """,
    
    """
    Initial state: At(C), Level(low), BoxAt(B), BananasAt(D)
    Goal state: Have(bananas)
    """,
    
    """
    Initial state: At(A), Level(high), BoxAt(C), BananasAt(B)
    Goal state: Have(bananas)
    """,
    
    """
    Initial state: At(D), Level(high), BoxAt(A), BananasAt(C)
    Goal state: Have(bananas)
    """
]

prompt_actions = """
Actions:
    // move from X to Y
    _Move(X, Y)_
        Preconditions: At(X), Level(low)
        Postconditions: not At(X), At(Y)

    // climb up on the box
    _ClimbUp(Location)_
        Preconditions: At(Location), BoxAt(Location), Level(low)
        Postconditions: Level(high), not Level(low)

    // climb down from the box
    _ClimbDown(Location)_
        Preconditions: At(Location), BoxAt(Location), Level(high)
        Postconditions: Level(low), not Level(high)

    // move monkey and box from X to Y
    _MoveBox(X, Y)_
        Preconditions: At(X), BoxAt(X), Level(low)
        Postconditions: BoxAt(Y), not BoxAt(X), At(Y), not At(X)

    // take the bananas
    _TakeBananas(Location)_
        Preconditions: At(Location), BananasAt(Location), Level(high)
        Postconditions: Have(bananas)
"""