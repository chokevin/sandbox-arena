"""Trading arena — evaluate trading strategies on historical price data."""

import json
import os

from arenas.base import Arena


# Simple inline price data so it works without external downloads
SAMPLE_SCENARIOS = [
    {"ticker": "AAPL", "prices": [150, 152, 148, 155, 160, 158, 162, 165, 163, 170, 168, 172, 175, 173, 178, 180, 176, 182, 185, 183]},
    {"ticker": "MSFT", "prices": [280, 285, 282, 290, 295, 288, 292, 298, 300, 305, 302, 310, 308, 315, 312, 320, 318, 325, 322, 330]},
    {"ticker": "GOOGL", "prices": [140, 138, 142, 145, 143, 148, 150, 147, 152, 155, 153, 158, 160, 157, 162, 165, 163, 168, 170, 172]},
    {"ticker": "AMZN", "prices": [180, 178, 182, 185, 183, 188, 190, 187, 185, 182, 180, 178, 175, 180, 185, 190, 195, 192, 198, 200]},
    {"ticker": "TSLA", "prices": [250, 260, 245, 270, 255, 280, 265, 290, 275, 295, 260, 285, 270, 300, 280, 310, 290, 305, 285, 320]},
]


class TradingArena(Arena):
    name = "trading"

    def scenarios(self) -> list[dict]:
        return SAMPLE_SCENARIOS

    def eval_script(self, agent_code: str, scenario: dict) -> str:
        prices_json = json.dumps(scenario["prices"])
        ticker = scenario["ticker"]

        return f'''
import json

# --- Agent code (untrusted) ---
{agent_code}
# --- End agent code ---

prices = {prices_json}
ticker = "{ticker}"
cash = 10000.0
position = 0
trade_log = []

for i in range(len(prices)):
    window = prices[:i+1]
    try:
        action = strategy(window, position, cash)
    except Exception as e:
        action = "hold"

    price = prices[i]
    if action == "buy" and cash >= price:
        shares = int(cash // price)
        if shares > 0:
            position += shares
            cash -= shares * price
            trade_log.append({{"day": i, "action": "buy", "shares": shares, "price": price}})
    elif action == "sell" and position > 0:
        cash += position * price
        trade_log.append({{"day": i, "action": "sell", "shares": position, "price": price}})
        position = 0

# Final portfolio value
final_value = cash + position * prices[-1]
total_return = (final_value - 10000.0) / 10000.0

# Simple Sharpe approximation (return / volatility)
if len(prices) > 1:
    daily_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    import statistics
    vol = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1.0
    sharpe = (total_return / len(prices) * 252) / (vol * (252 ** 0.5)) if vol > 0 else 0
else:
    sharpe = 0

result = {{
    "score": round(total_return * 100, 2),
    "passed": total_return > 0,
    "details": {{
        "ticker": ticker,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_approx": round(sharpe, 2),
        "trades": len(trade_log),
    }}
}}
print(json.dumps(result))
'''
