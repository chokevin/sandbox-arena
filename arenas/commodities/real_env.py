"""Real-data commodity trading Gym environment.

Uses actual historical OHLCV data from data/*.csv instead of simulated prices.
This is what alpha-transformer should train against.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Default commodities to trade
DEFAULT_SYMBOLS = ["gold", "oil", "wheat", "natgas"]

# Features to use from the CSV (pre-computed by fetch_data.py)
FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "rsi_14", "bb_position", "macd", "macd_signal",
]


def load_data(symbols: list[str], data_dir: str = None) -> dict[str, pd.DataFrame]:
    """Load CSV data for given symbols."""
    ddir = Path(data_dir) if data_dir else DATA_DIR
    data = {}
    for sym in symbols:
        path = ddir / f"{sym}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="date")
            data[sym] = df
    return data


class RealCommodityEnv(gym.Env):
    """Trade real commodity data.

    Same interface as CommodityTradingEnv but with historical prices.

    Observation per commodity (8 features from CSV):
        return_1d, return_5d, return_20d, volatility_20d,
        rsi_14, bb_position, macd, macd_signal
    Plus portfolio state (4): cash_pct, total_return, drawdown, days_remaining

    Action: continuous [-1, 1] per commodity (target position weight)
    """

    metadata = {"render_modes": []}

    def __init__(self, symbols: list[str] = None, data_dir: str = None,
                 window_days: int = 252, commission_bps: float = 2,
                 initial_cash: float = 100_000):
        super().__init__()

        self.symbols = symbols or DEFAULT_SYMBOLS
        self.all_data = load_data(self.symbols, data_dir)
        self.n_commodities = len(self.symbols)
        self.window_days = window_days
        self.commission_bps = commission_bps
        self.initial_cash = initial_cash

        if not self.all_data:
            raise ValueError(f"No data found. Run: python data/fetch_data.py")

        # Align dates across all symbols
        common_dates = None
        for sym, df in self.all_data.items():
            dates = set(df.index)
            common_dates = dates if common_dates is None else common_dates & dates
        self.common_dates = sorted(common_dates)

        obs_dim = self.n_commodities * len(FEATURE_COLS) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_commodities,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Random start point (leave room for window_days of trading)
        max_start = len(self.common_dates) - self.window_days - 1
        if max_start < 0:
            max_start = 0
        self.start_idx = int(rng.integers(0, max(max_start, 1)))
        self.dates = self.common_dates[self.start_idx:self.start_idx + self.window_days]
        self.day = 0
        self.cash = self.initial_cash
        self.positions = {sym: 0.0 for sym in self.symbols}
        self.portfolio_values = [self.initial_cash]
        self.peak_value = self.initial_cash
        self.trade_log = []

        return self._get_obs(), {}

    def _get_price(self, symbol: str, day_idx: int) -> float:
        date = self.dates[day_idx]
        return self.all_data[symbol].loc[date, "close"]

    def _get_features(self, symbol: str, day_idx: int) -> np.ndarray:
        date = self.dates[day_idx]
        row = self.all_data[symbol].loc[date]
        features = []
        for col in FEATURE_COLS:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            features.append(float(val))
        return np.array(features, dtype=np.float32)

    def _portfolio_value(self):
        val = self.cash
        for sym in self.symbols:
            val += self.positions[sym] * self._get_price(sym, self.day)
        return val

    def _get_obs(self):
        features = []
        for sym in self.symbols:
            features.append(self._get_features(sym, self.day))

        pv = self._portfolio_value()
        total_return = (pv / self.initial_cash) - 1
        drawdown = (self.peak_value - pv) / self.peak_value if self.peak_value > 0 else 0
        days_remaining = (len(self.dates) - self.day) / len(self.dates)

        portfolio_state = np.array([
            self.cash / pv if pv > 0 else 1.0,
            np.clip(total_return, -1, 5),
            np.clip(drawdown, 0, 1),
            days_remaining,
        ], dtype=np.float32)

        return np.concatenate(features + [portfolio_state])

    def step(self, action):
        action = np.clip(action, -1, 1)
        pv_before = self._portfolio_value()

        # Rebalance
        for i, sym in enumerate(self.symbols):
            target_weight = action[i] * (1.0 / self.n_commodities)
            target_value = pv_before * target_weight
            current_value = self.positions[sym] * self._get_price(sym, self.day)
            trade_value = target_value - current_value

            if abs(trade_value) > 10:
                price = self._get_price(sym, self.day)
                units = trade_value / price
                commission = abs(trade_value) * self.commission_bps / 10000
                self.positions[sym] += units
                self.cash -= trade_value + commission

                self.trade_log.append({
                    "day": self.day,
                    "date": self.dates[self.day],
                    "symbol": sym,
                    "units": round(units, 4),
                    "price": round(price, 2),
                    "value": round(trade_value, 2),
                })

        self.day += 1
        done = self.day >= len(self.dates) - 1

        pv_after = self._portfolio_value()
        self.portfolio_values.append(pv_after)
        self.peak_value = max(self.peak_value, pv_after)

        daily_return = (pv_after - pv_before) / pv_before if pv_before > 0 else 0
        drawdown = (self.peak_value - pv_after) / self.peak_value
        reward = daily_return * 100 - drawdown * 10

        info = {
            "portfolio_value": round(pv_after, 2),
            "total_return": round((pv_after / self.initial_cash - 1) * 100, 2),
            "drawdown": round(drawdown * 100, 2),
            "cash": round(self.cash, 2),
            "day": self.day,
            "date": self.dates[self.day] if not done else self.dates[-1],
        }

        if done:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            info["sharpe"] = round(sharpe, 3)
            info["final_value"] = round(pv_after, 2)
            info["n_trades"] = len(self.trade_log)
            reward += sharpe * 10

        return self._get_obs(), reward, done, False, info
