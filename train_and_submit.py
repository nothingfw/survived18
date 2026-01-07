# train_and_submit.py

import numpy as np

RANDOM_STATE = 42
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "submission.csv"

ID_COL = "apartment_id"
TARGET_COL = "survived_to_18jan"

# Категориальные признаки из условия
CAT_COLS = [
    "wing",
    "window_quality",
    "heating_type",
    "tree_species",
    "tree_form",
    "stand_type",
    "tinsel_level",
]


def main():
    # ---- читаем данные через pandas (это табличка, а не ASCII-арт) ----
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # базовые проверки
    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise ValueError(f"Не найден столбец {ID_COL} в train/test.")
    if TARGET_COL not in train.columns:
        raise ValueError(f"Не найден столбец {TARGET_COL} в train.csv.")

    # признаки
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]

    X = train[feature_cols].copy()
    y = train[TARGET_COL].astype(int).copy()

    X_test = test[feature_cols].copy()
    test_ids = test[ID_COL].copy()

    # какие категориальные реально присутствуют
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # приводим типы аккуратно
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("missing")
        X_test[c] = X_test[c].astype("string").fillna("missing")

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # split (стратификация, чтобы классы не перекосило)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ---- пробуем CatBoost: лучший вариант для такой задачи ----
    try:
        from catboost import CatBoostClassifier, Pool

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)

        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=200,
            od_type="Iter",
            od_wait=200,
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, val_pred)
        print(f"Validation ROC-AUC (CatBoost): {auc:.5f}")

        best_iter = model.get_best_iteration()
        if best_iter is None:
            best_iter = model.get_tree_count()
        else:
            best_iter = best_iter + 1  # у CatBoost best_iteration 0-based

        # финальная модель на всех данных с найденным числом итераций
        final_model = CatBoostClassifier(
            iterations=max(50, best_iter),
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        final_model.fit(Pool(X, y, cat_features=cat_cols))

        test_pred = final_model.predict_proba(X_test)[:, 1]

    except ImportError:
        # ---- fallback: sklearn (работает везде, но обычно слабее) ----
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

        clf = LogisticRegression(
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        model = Pipeline(steps=[("preprocess", pre), ("model", clf)])
        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, val_pred)
        print(f"Validation ROC-AUC (sklearn LR+OHE): {auc:.5f}")

        # дообучаем на всём train
        model.fit(X, y)
        test_pred = model.predict_proba(X_test)[:, 1]

    # страховка: вероятности должны быть [0,1] и без nan
    test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    # ---- submission ----
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: test_pred.astype(float),
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved: {SUBMISSION_PATH} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
