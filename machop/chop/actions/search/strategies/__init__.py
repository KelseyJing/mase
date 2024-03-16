# from .rl import StrategyRL
from .optuna import SearchStrategyOptuna
from .base import SearchStrategyBase
from .rl import StrategyRL #TODO: For RL based search strategy uncomment

SEARCH_STRATEGY_MAP = {
    "rl": StrategyRL, #TODO: For RL based search strategy uncomment
    "optuna": SearchStrategyOptuna,
}


def get_search_strategy_cls(name: str) -> SearchStrategyBase:
    if name not in SEARCH_STRATEGY_MAP:
        raise ValueError(
            f"{name} must be defined in {list(SEARCH_STRATEGY_MAP.keys())}."
        )
    return SEARCH_STRATEGY_MAP[name]
