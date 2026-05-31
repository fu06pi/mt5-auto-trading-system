from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Optional


@dataclasses.dataclass(frozen=True)
class Plan:
    action: str = "NONE"
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    fast_sma: float
    slow_sma: float
    momentum: float = 0.0
    score: float = 0.0
    signal: str = "NONE"


class BaseStrategyLogic:
    stop_loss_pts: Optional[float] = None

    def __init__(self, plan: Plan) -> None:
        self.plan = plan


class BaseRunner:
    def __init__(self) -> None:
        pass
