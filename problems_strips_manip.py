prompt_question="""
Task: Solve this robotic manipulation task.
"""

lst_tasks = [
    """
    Initial state: At(robot, A), OnTable(block1, B), OnTable(block2, C), OnTable(block3, D)
    Goal state: On(block1, block2)
    """,
    
    """
    Initial state: At(robot, D), OnTable(block1, A), OnTable(block2, B), OnTable(block3, C)
    Goal state: On(block2, block3)
    """,
    
    """
    Initial state: At(robot, B), OnTable(block1, D), OnTable(block2, A), OnTable(block3, C)
    Goal state: On(block3, block1)
    """,
    
    """
    Initial state: At(robot, C), OnTable(block1, B), OnTable(block2, D), OnTable(block3, A)
    Goal state: OnTable(block1, A), OnTable(block2, C), OnTable(block3, D)
    """,
    
    """
    Initial state: At(robot, A), OnTable(block1, C), OnTable(block2, B), On(block3, block1)
    Goal state: OnTable(block3, D)
    """,
    
    """
    Initial state: At(robot, D), OnTable(block1, A), On(block2, block3), OnTable(block3, C)
    Goal state: On(block1, block2)
    """
]

prompt_actions = """
Actions:
    // move robot from X to Y
    _Move(robot, X, Y)_
        Preconditions: At(robot, X)
        Postconditions: not At(robot, X), At(robot, Y)

    // pick up an object from location
    _PickUp(robot, Object, Location)_
        Preconditions: At(robot, Location), OnTable(Object, Location), HandEmpty(robot)
        Postconditions: Holding(robot, Object), not OnTable(Object, Location), not HandEmpty(robot)

    // place an object on location
    _Place(robot, Object, Location)_
        Preconditions: At(robot, Location), Holding(robot, Object)
        Postconditions: OnTable(Object, Location), not Holding(robot, Object), HandEmpty(robot)

    // stack an object on another object
    _Stack(robot, Object1, Object2)_
        Preconditions: Holding(robot, Object1), OnTable(Object2, Location)
        Postconditions: On(Object1, Object2), not Holding(robot, Object1), HandEmpty(robot)

    // unstack an object from another object
    _Unstack(robot, Object1, Object2)_
        Preconditions: At(robot, Location), On(Object1, Object2), HandEmpty(robot)
        Postconditions: Holding(robot, Object1), not On(Object1, Object2), HandEmpty(robot)
"""

