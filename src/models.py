"""The new algorithm (Explainable Boosting Machine) and the standard models it is compared with."""

from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import CAT_FEATURES, FEATURES, NUM_FEATURES


def ebm(params: dict, random_state: int, **hp) -> ExplainableBoostingRegressor:
    """EBM takes the raw frame: it bins numeric features and encodes `port` itself,
    and treats missing values as their own bin - no imputation, scaling or one-hot needed."""
    return ExplainableBoostingRegressor(
        feature_names=FEATURES,
        feature_types=["nominal"] + ["continuous"] * len(NUM_FEATURES),
        outer_bags=params["outer_bags"],
        inner_bags=params["inner_bags"],
        max_rounds=params["max_rounds"],
        early_stopping_rounds=params["early_stopping_rounds"],
        validation_size=params["validation_size"],
        random_state=random_state,
        n_jobs=-1,
        **hp,
    )


def _prep(scale: bool, only_port: bool = False):
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if only_port:
        return ColumnTransformer([("cat", onehot, CAT_FEATURES)])
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("cat", onehot, CAT_FEATURES), ("num", Pipeline(num), NUM_FEATURES)],
        verbose_feature_names_out=False,
    )


def baselines(random_state: int) -> dict:
    """Standard models. RF / XGBoost settings come from the earlier lecture activity
    (time-series CV search); the others are set for ~10,000 noisy daily rows."""
    return {
        "Port average": Pipeline([("prep", _prep(False, True)), ("model", LinearRegression())]),
        "Ridge regression": Pipeline([("prep", _prep(True)), ("model", Ridge(alpha=10.0))]),
        "k-NN (k=15)": Pipeline(
            [
                ("prep", _prep(True)),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
        "Decision tree": Pipeline(
            [
                ("prep", _prep(False)),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=8, min_samples_leaf=20, random_state=random_state
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", _prep(False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=16,
                        min_samples_leaf=12,
                        max_features=0.6,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", _prep(False)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.02,
                        max_depth=3,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        reg_lambda=5.0,
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
