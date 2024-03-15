import pandas as pd
import pandas_ta as ta


DEFAULT_STRATEGY = ta.Strategy(
    name="Default Strategy",
    description="",
    ta=[
        # {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        # {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        # {"kind": "macd", "fast": 8, "slow": 21},
        # {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
)

def add_technical_indicators(df):
    df.ta.strategy(DEFAULT_STRATEGY)
    max_length = 200
    return df, max_length
