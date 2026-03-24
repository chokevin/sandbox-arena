def strategy(prices, position, cash):
    """Buy and hold — buy on day 1, never sell. The baseline to beat."""
    if position == 0 and cash > 0:
        return "buy"
    return "hold"
