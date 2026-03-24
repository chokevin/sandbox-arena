def strategy(prices, position, cash):
    """Simple momentum strategy — buy when price is trending up, sell when down."""
    if len(prices) < 5:
        return "hold"
    if prices[-1] > prices[-5]:
        return "buy" if cash > 0 else "hold"
    else:
        return "sell" if position > 0 else "hold"
