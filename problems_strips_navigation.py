prompt_question="""
Task: Solve this indoor navigation task.
"""

lst_tasks = [
    """
    Initial state: At(robot, RoomA), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomB)
    """,
    
    """
    Initial state: At(robot, RoomD), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomA)
    """,
    
    """
    Initial state: At(robot, RoomB), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomC)
    """,
    
    """
    Initial state: At(robot, RoomC), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomD)
    """,
    
    """
    Initial state: At(robot, RoomA), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomD)
    """,
    
    """
    Initial state: At(robot, RoomD), DoorStatus(RoomA, RoomB, closed), DoorStatus(RoomB, RoomC, closed), DoorStatus(RoomC, RoomD, closed)
    Goal state: At(robot, RoomA)
    """
]

prompt_actions = """
Actions:
    // move robot from X to Y
    _Move(robot, X, Y)_
        Preconditions: At(robot, X), DoorStatus(X, Y, open)
        Postconditions: not At(robot, X), At(robot, Y)

    // open door between X and Y
    _OpenDoor(robot, X, Y)_
        Preconditions: At(robot, X), DoorStatus(X, Y, closed)
        Postconditions: DoorStatus(X, Y, open), not DoorStatus(X, Y, closed)

    // close door between X and Y
    _CloseDoor(robot, X, Y)_
        Preconditions: At(robot, X), DoorStatus(X, Y, open)
        Postconditions: DoorStatus(X, Y, closed), not DoorStatus(X, Y, open)
"""

