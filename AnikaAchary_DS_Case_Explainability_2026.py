"""
Supporting code for AnikaAchary_DS_Case_Analysis_2026.ipynb:
time-aware cross-validation + an explainability layer for the Ridge pipeline.

Works with the notebook's pipeline as-is:
    Pipeline([("prep", ColumnTransformer([("num", ..., NUMS), ("cat", OneHotEncoder(...), CATS)])),
              ("ridge", Ridge(...))])

Written with assistance from Claude.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit


# --------------------------------------------------------------------------------------
# Metrics: same as the notebook's report() -- predictions are rounded to whole days and
# clipped at 0 before scoring, because that's what actually gets submitted.
# --------------------------------------------------------------------------------------
def to_days(pred):
    return np.clip(np.round(pred), 0, None)


def rounded_mae(y_true, y_pred):
    return np.mean(np.abs(to_days(y_pred) - np.asarray(y_true)))


ROUNDED_MAE = make_scorer(rounded_mae, greater_is_better=False)


# --------------------------------------------------------------------------------------
# Splitter: forward-chaining CV, purged by kit_end
# --------------------------------------------------------------------------------------
def purged_time_splits(df, start_col="kit_start", end_col="kit_end", n_splits=5, max_train_size=None):
    """TimeSeriesSplit (expanding window, or sliding if max_train_size is set), then drop any
    training order whose kit_end is on/after the first day of the validation block -- its
    label wouldn't be known yet at prediction time. df must already be sorted by start_col.
    Returns a list of (train_idx, val_idx) that GridSearchCV accepts as cv=."""
    starts = df[start_col].to_numpy()
    ends = df[end_col].to_numpy()
    splits = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size).split(df):
        val_start = starts[va_idx[0]]
        keep = ends[tr_idx] < val_start
        # orders that START on the boundary day get split across train/val by row count;
        # the end-date rule above already removes them from train
        splits.append((tr_idx[keep], va_idx))
    return splits


def describe_splits(df, splits, start_col="kit_start"):
    rows = []
    for i, (tr, va) in enumerate(splits):
        rows.append({
            "fold": i,
            "train_rows": len(tr),
            "train_from": df[start_col].iloc[tr[0]].date(),
            "train_to": df[start_col].iloc[tr[-1]].date(),
            "val_rows": len(va),
            "val_from": df[start_col].iloc[va[0]].date(),
            "val_to": df[start_col].iloc[va[-1]].date(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Small helpers to reach inside the pipeline
# --------------------------------------------------------------------------------------
def _parts(pipe):
    """(preprocessor, final estimator) regardless of what the steps are named."""
    return pipe.steps[0][1], pipe.steps[-1][1]


def numeric_coefs_days_per_unit(pipe, num_features):
    """Coefficients of the numeric features in ORIGINAL units (days per +1 unit of the
    feature, after make_features). Comparable across refits even though each refit has
    its own StandardScaler. (order_amt is log1p-transformed, so its unit is +1 log-dollar.)"""
    prep, est = _parts(pipe)
    num_pipe = prep.named_transformers_["num"]
    scaler = num_pipe[-1] if hasattr(num_pipe, "steps") else num_pipe
    return pd.Series(est.coef_[: len(num_features)] / scaler.scale_, index=num_features)


# --------------------------------------------------------------------------------------
# Explainer
# --------------------------------------------------------------------------------------
class RidgeExplainer:
    """Additive explanations for a fitted (ColumnTransformer -> linear model) pipeline.

    prediction = baseline + sum_j contribution_j, exactly, where
        contribution_j = coef_j * (x_j - mean_j)   in the transformed (scaled / one-hot) space
        baseline       = average raw prediction over the background (training) data
    One-hot columns are summed back to their original feature, so family_desc is one number.
    For a linear model this is the same as SHAP's LinearExplainer -- no extra library needed.
    All contributions are in DAYS.
    """

    def __init__(self, pipe, X_background, num_features, cat_features):
        self.pipe = pipe
        self.prep, self.est = _parts(pipe)
        self.num, self.cat = list(num_features), list(cat_features)
        self.features = self.num + self.cat

        coef = np.asarray(self.est.coef_).ravel()
        Xt_bg = self.prep.transform(X_background[self.features])
        mean_t = np.asarray(Xt_bg.mean(axis=0)).ravel()

        # group matrix: transformed column -> original feature
        enc = self.prep.named_transformers_["cat"]
        group_of_col = list(range(len(self.num)))
        for k, cats in enumerate(enc.categories_):
            group_of_col += [len(self.num) + k] * len(cats)
        G = sparse.csr_matrix((np.ones(len(group_of_col)), (np.arange(len(group_of_col)), group_of_col)),
                              shape=(len(group_of_col), len(self.features)))

        self._W = sparse.diags(coef) @ G                       # cols -> grouped, weighted
        self._offset = np.asarray(((mean_t * coef) @ G)).ravel()
        self.baseline = float(self.est.intercept_ + mean_t @ coef)
        self.coef_names = self.prep.get_feature_names_out()
        self.coef = coef
        self._mean_t = mean_t

    # ---------------- core ----------------
    def contributions(self, X):
        """DataFrame (orders x original features) of contributions in days."""
        Xt = self.prep.transform(X[self.features])
        raw = Xt @ self._W
        raw = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
        return pd.DataFrame(raw - self._offset, index=X.index, columns=self.features)

    # ---------------- 1) individual order ----------------
    def explain_order(self, x_row, actual=None, ax=None, title=None, top=None):
        """Waterfall for one order. x_row: a one-row DataFrame (or Series) of make_features output."""
        if isinstance(x_row, pd.Series):
            x_row = x_row.to_frame().T
        c = self.contributions(x_row).iloc[0]
        raw_pred = self.baseline + c.sum()
        assert np.isclose(raw_pred, self.pipe.predict(x_row[self.features])[0]), "contributions must sum to prediction"

        c = c.reindex(c.abs().sort_values(ascending=False).index)
        if top is not None and len(c) > top:
            rest = c.iloc[top:].sum()
            c = pd.concat([c.iloc[:top], pd.Series({"(all other features)": rest})])
        values = [x_row.iloc[0][f] if f in x_row.columns else "" for f in c.index]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 0.35 * len(c) + 1.5))
        running = self.baseline
        for i, v in enumerate(c.values):
            ax.barh(i, v, left=running, color="#c0504d" if v > 0 else "#4f81bd")
            running += v
        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:g}" if float(v).is_integer() else f"{v:.2f}"
            return str(v)[:22]
        labels = [f"{f} = {fmt(val)}" if val != "" else f for f, val in zip(c.index, values)]
        ax.set_yticks(range(len(c)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(self.baseline, color="grey", ls="--", lw=0.8, label=f"baseline {self.baseline:.2f}")
        ax.axvline(raw_pred, color="black", lw=1, label=f"model {raw_pred:.2f} -> {to_days(raw_pred):.0f}d")
        if actual is not None:
            ax.axvline(actual, color="green", lw=1, label=f"actual {actual:.0f}d")
        ax.margins(x=0.1)
        ax.set_xlabel("kit duration (days)")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(title or "Order explanation", fontsize=10)

        return pd.DataFrame({"value": values, "contribution_days": c.values}, index=c.index)

    # ---------------- 2) aggregate ----------------
    def global_importance(self, X, y=None, n_repeats=5, random_state=0, max_rows=20000):
        """Mean |contribution| per feature (days), plus permutation importance (increase in
        rounded MAE when the feature is shuffled) when labels are given."""
        contrib = self.contributions(X)
        out = pd.DataFrame({
            "mean_abs_contribution_days": contrib.abs().mean(),
            "share_of_total": contrib.abs().mean() / contrib.abs().mean().sum(),
        })
        if y is not None:
            Xs, ys = X[self.features], y
            if len(Xs) > max_rows:
                idx = np.random.default_rng(random_state).choice(len(Xs), max_rows, replace=False)
                Xs, ys = Xs.iloc[idx], y.iloc[idx]
            perm = permutation_importance(self.pipe, Xs, ys, scoring=ROUNDED_MAE,
                                          n_repeats=n_repeats, random_state=random_state)
            out["perm_importance_mae_increase"] = pd.Series(perm.importances_mean, index=self.features)
        return out.sort_values("mean_abs_contribution_days", ascending=False)

    def level_effects(self, feature, X=None, min_orders=0):
        """Per-category effect (days vs. the average order) for one categorical feature,
        with how many orders in X have that level (levels with < min_orders are dropped)."""
        prefix = f"cat__{feature}_"
        mask = np.array([n.startswith(prefix) for n in self.coef_names])
        eff = pd.Series(self.coef[mask], index=[n[len(prefix):] for n in self.coef_names[mask]])
        # center on the frequency-weighted average level (same reference as contributions())
        eff = eff - self._mean_t[mask] @ self.coef[mask]
        out = eff.rename("effect_days").to_frame()
        if X is not None:
            out["n_orders"] = X[feature].value_counts().reindex(out.index).fillna(0).astype(int)
            out = out[out["n_orders"] >= min_orders]
        return out.sort_values("effect_days")

    # ---------------- 3) across time (on a fixed model) ----------------
    def contributions_by_period(self, X, dates, freq="W", signed=True):
        """Average contribution per feature per period. signed=True shows direction of push,
        signed=False shows how much each feature is driving predictions in that period."""
        contrib = self.contributions(X)
        if not signed:
            contrib = contrib.abs()
        period = pd.Series(pd.to_datetime(dates).values, index=X.index).dt.to_period(freq)
        return contrib.groupby(period).mean()


# --------------------------------------------------------------------------------------
# Across time (refitting): how the model's reasoning changes over the timeline
# --------------------------------------------------------------------------------------
def rolling_refit(pipe, X, y, dates, num_features, cat_features, window_days=60, step_days=14):
    """Refit a clone of `pipe` on rolling windows of the timeline. For each window returns
      - numeric coefficients in days-per-unit (comparable across windows)
      - mean |contribution| per feature on that window's orders (comparable for every feature)
    Descriptive only: this shows drift in the process, it is NOT a model evaluation."""
    dates = pd.to_datetime(pd.Series(np.asarray(dates), index=X.index))
    win, step = pd.Timedelta(days=window_days), pd.Timedelta(days=step_days)
    t0, t_end = dates.min(), dates.max()
    coef_rows, imp_rows = [], []
    while t0 + win <= t_end + pd.Timedelta(days=1):
        m = (dates >= t0) & (dates < t0 + win)
        p = clone(pipe).fit(X[m], y[m])
        mid = (t0 + win / 2).normalize()
        coef_rows.append(numeric_coefs_days_per_unit(p, num_features).rename(mid))
        ex = RidgeExplainer(p, X[m], num_features, cat_features)
        imp_rows.append(ex.contributions(X[m]).abs().mean().rename(mid))
        t0 += step
    return pd.DataFrame(coef_rows), pd.DataFrame(imp_rows)
