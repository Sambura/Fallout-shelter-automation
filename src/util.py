H_DIRECTIONS: list[str] = ['left', 'right']
V_DIRECTIONS: list[str] = ['up', 'down']

def are_opposite_directions(direction: str, other_direction: str) -> bool:
    if set([direction, other_direction]) == set(H_DIRECTIONS): return True
    if set([direction, other_direction]) == set(V_DIRECTIONS): return True
    return False

def is_vertical(direction: str) -> bool: return direction in V_DIRECTIONS
def is_horizontal(direction: str) -> bool: return direction in H_DIRECTIONS
