def survive(health, food, energy, shelter, turn):
    """Balanced survival agent — prioritizes based on most urgent need.

    Args:
        health: current health (0-100)
        food: current food supply (can go negative)
        energy: current energy (0-100)
        shelter: shelter level (0-100)
        turn: current turn number
    Returns:
        "forage", "rest", "explore", or "build"
    """
    # Critical: rest if about to die
    if health < 30 or energy < 10:
        return "rest"

    # Build shelter early for storm protection
    if shelter < 40 and energy > 30 and food > 20:
        return "build"

    # Forage if food is low
    if food < 25:
        return "forage"

    # Explore if we're comfortable
    if food > 40 and energy > 40 and health > 60:
        return "explore"

    # Default: forage
    return "forage"
