import pandas as pd

CATEGORICAL_FEATURES = [
    "Cuisinetype",
    "Segment",
    "Subregion"
]

NUMERIC_FEATURES = [
    "GrowthFactor",
    "AOV",
    "MonthlyOrders",
    "InStoreOrders",
    "UberEatsOrders",
    "DoorDashOrders",
    "SelfDeliveryOrders",
    "COGSRate",
    "OPEXRate",
    "CommissionRate",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",
]

FEATURE_COLUMN = CATEGORICAL_FEATURES + NUMERIC_FEATURES

PROFIT_COLUMNS = [
    "InStoreNetProfit",
    "UberEatsNetProfit",
    "DoorDashNetProfit",
    "SelfDeliveryNetProfit",
]

TARGET_COLUMN = "TotalNetProfit"

def bulid_features(df: pd.DataFrame):
    """Build model inputs and the monthly total-profit target."""
    required = FEATURE_COLUMN + PROFIT_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    X = df[FEATURE_COLUMNS ].copy()
    
    y = df[PROFIT_COLUMNS].sum(axis=1)
    y.name = TARGET_COLUMN
    
    return X, y