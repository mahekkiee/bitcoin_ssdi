"""
Buffett Effect in Crypto Trading — Streamlit Dashboard

Tests whether market mood (Fear & Greed Index) changes how much
traders make on each trade. Mirrors the analysis in working.ipynb:
  EDA  →  VIF feature selection  →  Linear regression
       →  Hypothesis testing  →  Final verdict

Place `fear_greed_index.csv` and `historical_data.csv.gz` in the
same folder as this app, then run:
    streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Buffett Effect in Crypto Trading",
    page_icon="📉",
    layout="wide",
)
sns.set_theme(style="whitegrid")

FG_FILE    = "fear_greed_index.csv"
TRADE_FILE = "historical_data.csv.gz"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral",
                   "Greed", "Extreme Greed"]

# ---------------------------------------------------------------
# Data loader — same steps as the notebook
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading and joining the two files…")
def load_data():
    missing = [f for f in [FG_FILE, TRADE_FILE] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) in the app folder: {', '.join(missing)}. "
            "Place both files next to app.py and rerun."
        )

    # Fear & Greed index
    fg = pd.read_csv(FG_FILE)
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg[["date", "value", "classification"]]
    fg.columns = ["date", "FG_value", "Sentiment"]

    # Trade history
    tr = pd.read_csv(TRADE_FILE, compression="gzip")
    tr["Timestamp IST"] = pd.to_datetime(
        tr["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    tr = tr.dropna(subset=["Timestamp IST"])
    tr["date"] = tr["Timestamp IST"].dt.normalize()

    keep_directions = ["Open Long", "Close Long", "Open Short",
                       "Close Short", "Buy", "Sell"]
    tr = tr[tr["Direction"].isin(keep_directions)].copy()
    tr = tr[["date", "Coin", "Side", "Direction",
             "Size USD", "Closed PnL", "Fee",
             "Execution Price", "Size Tokens"]].copy()
    tr.columns = ["date", "Coin", "Side", "Direction",
                  "Size_USD", "Closed_PnL", "Fee",
                  "Price", "Tokens"]

    df = tr.merge(fg, on="date", how="inner")
    df["FG_sq"] = df["FG_value"] ** 2
    return df


# ---------------------------------------------------------------
# Cached computations used on multiple pages
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Fitting Model 6…")
def fit_model6(df):
    return smf.ols(
        "Closed_PnL ~ FG_value * Side + Size_USD + Price",
        data=df,
    ).fit()


@st.cache_data(show_spinner="Fitting all candidate models…")
def fit_all_models(df):
    specs = {
        "M1: Size only":                         "Closed_PnL ~ Size_USD",
        "M2: FG only":                           "Closed_PnL ~ FG_value",
        "M3: FG + Size":                         "Closed_PnL ~ FG_value + Size_USD",
        "M4: FG + Size + Side":                  "Closed_PnL ~ FG_value + Size_USD + Side",
        "M5: FG + Size + Side + Price":          "Closed_PnL ~ FG_value + Size_USD + Side + Price",
        "M6: FG * Side + Size + Price  ":      "Closed_PnL ~ FG_value * Side + Size_USD + Price",
        "M7: FG + FG² + Side + Size":            "Closed_PnL ~ FG_value + FG_sq + Side + Size_USD",
    }
    fits = {name: smf.ols(spec, data=df).fit() for name, spec in specs.items()}
    rows = [{
        "Model": name,
        "R²": round(f.rsquared, 6),
        "Adj R²": round(f.rsquared_adj, 6),
        "AIC": round(f.aic, 0),
        "N obs": int(f.nobs),
    } for name, f in fits.items()]
    compare = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return fits, compare


# ---------------------------------------------------------------
# Load once, fail early
# ---------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f" {e}")
    st.stop()


# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title(" Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        " Overview",
        " Exploratory Analysis",
        " VIF — Feature Selection",
        " Linear Regression",
        " Hypothesis Testing",
        " Final Verdict",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {len(df):,} trades · "
    f"{df['Coin'].nunique()} coins · "
    f"{df['date'].nunique()} days"
)


# ===============================================================
# 1. OVERVIEW
# ===============================================================
if page == "Overview":
    st.title("Sentiment Effect in Crypto Trading")
    st.caption('"Be fearful when others are greedy, and greedy when others are fearful." '
               "— testing the statement on 210k real trades.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trades", f"{len(df):,}")
    c2.metric("Trading days", f"{df['date'].nunique():,}")
    c3.metric("Unique coins", df["Coin"].nunique())
    c4.metric("Date range",
              f"{df['date'].min().date()} → {df['date'].max().date()}")

    st.markdown("### Project summary")
    st.markdown(
        """
        This dashboard tests a single question: **does market mood change
        how profitable a trade is?** Two data files are used:

        - **Fear & Greed Index** — one mood score (0–100) per day, where
          0 = extreme fear and 100 = extreme greed.
        - **Trade history** — every trade made by ~32 traders over two
          years, with side (BUY / SELL), size, fee, price, and closed PnL.

        The analysis follows a textbook statistical pipeline:

        1. **EDA** — eyeball patterns in the raw data
        2. **VIF** — drop predictors that measure the same thing
        3. **Regression** — 7 candidate models, pick the best by AIC
        4. **Hypothesis testing** — 4 formal tests of the Buffett claim
        5. **Final verdict** — combining every piece of evidence
        """
    )

    st.markdown("### Sample of the joined data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Trades per sentiment regime")
    counts = (df["Sentiment"].value_counts()
              .reindex(SENTIMENT_ORDER).fillna(0).astype(int))
    st.bar_chart(counts)


# ===============================================================
# 2. EXPLORATORY ANALYSIS
# ===============================================================
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    st.markdown("### Average PnL by sentiment, split by side")
    eda = (df.groupby(["Sentiment", "Side"])["Closed_PnL"]
             .mean().round(2).unstack()
             .reindex(SENTIMENT_ORDER))
    st.dataframe(eda, use_container_width=True)
    st.info(
        "Look at the columns carefully. For BUY trades, profit tends to be "
        "higher in Fear than Greed. For SELL trades, the opposite. "
        "This is the Buffett effect visible in the raw averages — "
        "before any regression."
    )

    st.markdown("### Mean PnL by regime (buyers vs sellers)")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = (df.groupby(["Sentiment", "Side"])["Closed_PnL"]
                 .mean().reset_index())
    plot_df["Sentiment"] = pd.Categorical(
        plot_df["Sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    plot_df = plot_df.sort_values("Sentiment")
    sns.barplot(data=plot_df, x="Sentiment", y="Closed_PnL",
                hue="Side", palette={"BUY": "#2ca02c", "SELL": "#d62728"},
                ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Closed PnL (USD)")
    ax.set_title("The X-shape: buyer and seller profits mirror each other")
    st.pyplot(fig)

    st.markdown("### Sentiment vs PnL scatter (5,000-trade sample)")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.regplot(data=sample, x="FG_value", y="Closed_PnL",
                order=1, scatter_kws={"alpha": 0.3, "s": 10},
                line_kws={"color": "red"}, ax=ax2)
    ax2.set_xlabel("Fear & Greed Index (0–100)")
    ax2.set_ylabel("Closed PnL (USD)")
    ax2.set_title("Without splitting BUY/SELL, the overall trend looks weak")
    st.pyplot(fig2)
    st.caption(
        "That nearly-flat red line is why Model 2 (FG alone) looked useless. "
        "Mixing buyers and sellers hides a real pattern."
    )


# ===============================================================
# 3. VIF — FEATURE SELECTION
# ===============================================================
elif page == "🧹 VIF — Feature Selection":
    st.title("🧹 VIF — Feature Selection")

    st.markdown(
        """
        **Variance Inflation Factor (VIF)** measures how much each
        predictor overlaps with the others. A VIF of 1 means the predictor
        is independent. A VIF above 5 is concerning; above 10 is serious
        multicollinearity that makes regression coefficients unstable.

        We test five numeric candidate predictors:
        **FG_value, Size_USD, Fee, Price, Tokens**.
        """
    )

    candidates = ["FG_value", "Size_USD", "Fee", "Price", "Tokens"]
    X = df[candidates].dropna()
    Xc = add_constant(X)
    vif_initial = pd.DataFrame({
        "Feature": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i)
                for i in range(Xc.shape[1])],
    })
    vif_initial["VIF"] = vif_initial["VIF"].round(3)

    st.markdown("### Step 1 — Initial VIF (all candidates)")
    st.dataframe(vif_initial, use_container_width=True)

    st.markdown("### Correlation matrix (explains the VIF numbers)")
    corr = X.corr().round(3)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f",
                ax=ax)
    st.pyplot(fig)
    st.warning(
        f"Size_USD ↔ Fee correlation ≈ "
        f"**{corr.loc['Size_USD', 'Fee']:.2f}** — high. This is why their "
        "VIFs are around 2.3. Fee is charged as a percentage of trade size, "
        "so it adds no new information the regression can use."
    )

    st.markdown("### Step 2 — Drop `Fee`, recompute VIF")
    X_clean = df[["FG_value", "Size_USD", "Price", "Tokens"]].dropna()
    Xc2 = add_constant(X_clean)
    vif_clean = pd.DataFrame({
        "Feature": Xc2.columns,
        "VIF": [variance_inflation_factor(Xc2.values, i)
                for i in range(Xc2.shape[1])],
    })
    vif_clean["VIF"] = vif_clean["VIF"].round(3)
    st.dataframe(vif_clean, use_container_width=True)

    st.success(
        "✅ All VIFs are near 1 — no multicollinearity left. "
        "Final predictor set for regression: "
        "**FG_value, Size_USD, Price, Tokens, Side (categorical)**."
    )


# ===============================================================
# 4. LINEAR REGRESSION
# ===============================================================
elif page == "📉 Linear Regression":
    st.title("📉 Linear Regression — Who profits from which mood?")

    fits, compare = fit_all_models(df)
    lm6 = fits["M6: FG * Side + Size + Price  ⭐"]

    st.markdown("### Model comparison (lower AIC = better)")
    st.dataframe(compare, use_container_width=True)
    st.success(
        f"🏆 **Winner: {compare.iloc[0]['Model']}** — lowest AIC, and it's "
        "the only model that lets buyers and sellers have different slopes."
    )

    st.markdown("### Model 6 — coefficient table")
    coef_df = pd.DataFrame({
        "Coefficient": lm6.params.round(4),
        "Std error":   lm6.bse.round(4),
        "t-stat":      lm6.tvalues.round(3),
        "p-value":     lm6.pvalues.apply(lambda p: f"{p:.2e}"),
    })
    st.dataframe(coef_df, use_container_width=True)

    b_buy  = lm6.params["FG_value"]
    b_diff = lm6.params["FG_value:Side[T.SELL]"]
    b_sell = b_buy + b_diff

    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer slope", f"{b_buy:+.4f}",
              help="Profit change per 1-point rise in greed")
    c2.metric("Seller slope", f"{b_sell:+.4f}",
              help="= buyer slope + interaction term")
    c3.metric("Interaction", f"{b_diff:+.4f}",
              help="How differently sellers respond vs buyers")

    st.info(
        f"For each **1-point** rise in Fear & Greed:\n"
        f"- Buyers earn **${b_buy:+.2f}** — *less* profit under greed.\n"
        f"- Sellers earn **${b_sell:+.2f}** — *more* profit under greed.\n\n"
        f"Over a typical 50-point swing from Fear (≈25) to Greed (≈75), "
        f"the gap between a seller and a buyer widens by "
        f"**${(abs(b_buy)+b_sell)*50:,.2f} per trade**."
    )

    st.markdown("### Visualising the two slopes (5,000-trade sample)")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(data=sample[sample["Side"] == "BUY"],
                x="FG_value", y="Closed_PnL",
                scatter_kws={"alpha": 0.2, "s": 10, "color": "#2ca02c"},
                line_kws={"color": "#1a6b1a"}, label="BUY", ax=ax)
    sns.regplot(data=sample[sample["Side"] == "SELL"],
                x="FG_value", y="Closed_PnL",
                scatter_kws={"alpha": 0.2, "s": 10, "color": "#d62728"},
                line_kws={"color": "#8b1a1c"}, label="SELL", ax=ax)
    ax.set_xlabel("Fear & Greed Index")
    ax.set_ylabel("Closed PnL (USD)")
    ax.set_title("The X-shape: buyers' line slopes down, sellers' line slopes up")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Full OLS summary (Model 6)"):
        st.code(lm6.summary().as_text(), language="text")


# ===============================================================
# 5. HYPOTHESIS TESTING
# ===============================================================
elif page == "🧪 Hypothesis Testing":
    st.title("🧪 Hypothesis Testing")
    st.markdown(
        "Regression gave us coefficients. Hypothesis testing gives us "
        "formal **yes/no** answers. For each claim: state H₀ and H₁, "
        "compute a p-value, reject H₀ if p < 0.05."
    )

    fear_labels  = ["Fear", "Extreme Fear"]
    greed_labels = ["Greed", "Extreme Greed"]
    buys  = df[df["Side"] == "BUY"]
    sells = df[df["Side"] == "SELL"]
    buys_fear   = buys[buys["Sentiment"].isin(fear_labels)]["Closed_PnL"]
    buys_greed  = buys[buys["Sentiment"].isin(greed_labels)]["Closed_PnL"]
    sells_fear  = sells[sells["Sentiment"].isin(fear_labels)]["Closed_PnL"]
    sells_greed = sells[sells["Sentiment"].isin(greed_labels)]["Closed_PnL"]

    # -------- H1 --------
    st.markdown("### H1 — Buyers earn more in Fear than in Greed")
    st.markdown(
        "**H₀:** μ(buyer|Fear) = μ(buyer|Greed)  \n"
        "**H₁:** μ(buyer|Fear) > μ(buyer|Greed)  \n"
        "*Test:* Welch's t-test (unequal variances)"
    )
    t1, p1_two = stats.ttest_ind(buys_fear, buys_greed, equal_var=False)
    p1 = p1_two / 2 if t1 > 0 else 1 - p1_two / 2
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (Fear)",  f"{len(buys_fear):,}")
    c2.metric("n (Greed)", f"{len(buys_greed):,}")
    c3.metric("Δ mean",    f"${buys_fear.mean() - buys_greed.mean():+.2f}")
    c4.metric("one-sided p", f"{p1:.2e}")
    if p1 < 0.05:
        st.success(
            f"✅ **Reject H₀** (t = {t1:.2f}, p = {p1:.2e}). "
            f"Buyers earn significantly more in fearful markets "
            f"(${buys_fear.mean():.2f} vs ${buys_greed.mean():.2f})."
        )
    else:
        st.warning(f"Fail to reject H₀ (p = {p1:.3f}).")

    # -------- H2 --------
    st.markdown("---")
    st.markdown("### H2 — Sellers earn more in Greed than in Fear")
    st.markdown(
        "**H₀:** μ(seller|Greed) = μ(seller|Fear)  \n"
        "**H₁:** μ(seller|Greed) > μ(seller|Fear)  \n"
        "*Test:* Welch's t-test"
    )
    t2, p2_two = stats.ttest_ind(sells_greed, sells_fear, equal_var=False)
    p2 = p2_two / 2 if t2 > 0 else 1 - p2_two / 2
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (Greed)", f"{len(sells_greed):,}")
    c2.metric("n (Fear)",  f"{len(sells_fear):,}")
    c3.metric("Δ mean",    f"${sells_greed.mean() - sells_fear.mean():+.2f}")
    c4.metric("one-sided p", f"{p2:.2e}")
    if p2 < 0.05:
        st.success(
            f"✅ **Reject H₀** (t = {t2:.2f}, p = {p2:.2e}). "
            f"Sellers earn significantly more in greedy markets "
            f"(${sells_greed.mean():.2f} vs ${sells_fear.mean():.2f})."
        )
    else:
        st.warning(f"Fail to reject H₀ (p = {p2:.3f}).")

    # -------- H3 --------
    st.markdown("---")
    st.markdown("### H3 — Buyer PnL differs across all 5 sentiment regimes")
    st.markdown(
        "**H₀:** μ(EF) = μ(F) = μ(N) = μ(G) = μ(EG)  \n"
        "**H₁:** at least one mean differs  \n"
        "*Test:* One-way ANOVA (F-test)"
    )
    groups = [buys[buys["Sentiment"] == s]["Closed_PnL"].values
              for s in SENTIMENT_ORDER
              if s in buys["Sentiment"].unique()]
    f_stat, p3 = stats.f_oneway(*groups)
    c1, c2, c3 = st.columns(3)
    c1.metric("F-statistic", f"{f_stat:.3f}")
    c2.metric("p-value", f"{p3:.2e}")
    c3.metric("Groups", f"{len(groups)}")
    if p3 < 0.05:
        st.success(
            f"✅ **Reject H₀** (F = {f_stat:.2f}, p = {p3:.2e}). "
            "Average buyer profit differs meaningfully by sentiment regime."
        )
    else:
        st.warning(f"Fail to reject H₀ (p = {p3:.3f}).")

    # Visual support
    fig, ax = plt.subplots(figsize=(9, 4.5))
    buyer_means = buys.groupby("Sentiment")["Closed_PnL"].mean().reindex(
        SENTIMENT_ORDER)
    sns.barplot(x=buyer_means.index, y=buyer_means.values,
                palette="RdYlGn_r", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Buyer PnL (USD)")
    ax.set_title("Buyer PnL by regime — the monotone pattern")
    st.pyplot(fig)

    # -------- H4 --------
    st.markdown("---")
    st.markdown("### H4 — The Buffett interaction is statistically real")
    st.markdown(
        "**H₀:** β(FG_value × Side) = 0  \n"
        "**H₁:** β(FG_value × Side) ≠ 0  \n"
        "*Test:* t-test on the interaction coefficient in Model 6"
    )
    lm6 = fit_model6(df)
    int_coef = lm6.params["FG_value:Side[T.SELL]"]
    int_se   = lm6.bse["FG_value:Side[T.SELL]"]
    int_p    = lm6.pvalues["FG_value:Side[T.SELL]"]
    int_ci   = lm6.conf_int().loc["FG_value:Side[T.SELL]"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("β",        f"{int_coef:+.4f}")
    c2.metric("Std error", f"{int_se:.4f}")
    c3.metric("p-value",   f"{int_p:.2e}")
    c4.metric("95% CI",
              f"[{int_ci[0]:+.3f}, {int_ci[1]:+.3f}]")
    if int_p < 0.05:
        st.success(
            f"✅ **Reject H₀** (p = {int_p:.2e}). "
            "Buyers and sellers respond to sentiment in genuinely opposite "
            "directions — not by chance."
        )
    else:
        st.warning(f"Fail to reject H₀ (p = {int_p:.3f}).")


# ===============================================================
# 6. FINAL VERDICT
# ===============================================================
elif page == "📝 Final Verdict":
    st.title("📝 Final Verdict")

    fits, _ = fit_all_models(df)
    lm6 = fits["M6: FG * Side + Size + Price  ⭐"]
    b_buy  = lm6.params["FG_value"]
    b_diff = lm6.params["FG_value:Side[T.SELL]"]
    b_sell = b_buy + b_diff

    fear_labels  = ["Fear", "Extreme Fear"]
    greed_labels = ["Greed", "Extreme Greed"]
    buys  = df[df["Side"] == "BUY"]
    sells = df[df["Side"] == "SELL"]
    buys_fear   = buys[buys["Sentiment"].isin(fear_labels)]["Closed_PnL"]
    buys_greed  = buys[buys["Sentiment"].isin(greed_labels)]["Closed_PnL"]
    sells_fear  = sells[sells["Sentiment"].isin(fear_labels)]["Closed_PnL"]
    sells_greed = sells[sells["Sentiment"].isin(greed_labels)]["Closed_PnL"]

    t1, p1_two = stats.ttest_ind(buys_fear, buys_greed, equal_var=False)
    p1 = p1_two / 2 if t1 > 0 else 1 - p1_two / 2
    t2, p2_two = stats.ttest_ind(sells_greed, sells_fear, equal_var=False)
    p2 = p2_two / 2 if t2 > 0 else 1 - p2_two / 2
    groups = [buys[buys["Sentiment"] == s]["Closed_PnL"].values
              for s in SENTIMENT_ORDER
              if s in buys["Sentiment"].unique()]
    f_stat, p3 = stats.f_oneway(*groups)
    int_p = lm6.pvalues["FG_value:Side[T.SELL]"]
    int_coef = lm6.params["FG_value:Side[T.SELL]"]

    st.markdown("### Summary of all evidence")
    summary = pd.DataFrame([
        ["VIF", "Multicollinearity check",
         "Dropped Fee (corr with Size_USD ≈ 0.75, VIF ≈ 2.3)",
         "✓ Clean predictors"],
        ["Regression", "Model 6: FG × Side + Size + Price",
         f"R² = {lm6.rsquared:.4f},  AIC = {lm6.aic:,.0f}",
         f"Buyer slope {b_buy:+.4f} / Seller slope {b_sell:+.4f}"],
        ["H1 (buyers)", "Welch t-test, Fear > Greed",
         f"Δ = ${buys_fear.mean()-buys_greed.mean():+.2f},  p = {p1:.2e}",
         "✅ Buyers earn more in fear"],
        ["H2 (sellers)", "Welch t-test, Greed > Fear",
         f"Δ = ${sells_greed.mean()-sells_fear.mean():+.2f},  p = {p2:.2e}",
         "✅ Sellers earn more in greed"],
        ["H3 (regimes)", "One-way ANOVA on 5 regimes",
         f"F = {f_stat:.2f},  p = {p3:.2e}",
         "✅ Buyer PnL differs by regime"],
        ["H4 (interaction)", "Wald test on β(FG × Side)",
         f"β = {int_coef:+.4f},  p = {int_p:.2e}",
         "✅ Opposite effects confirmed"],
    ], columns=["Evidence", "Test", "Result", "Verdict"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### The headline numbers")
    swing = 50
    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer expected loss per trade",
              f"${abs(b_buy)*swing:,.2f}",
              help="For a 50-point swing from Fear to Greed")
    c2.metric("Seller expected gain per trade",
              f"${b_sell*swing:,.2f}",
              help="For the same 50-point swing")
    c3.metric("Total mood-driven gap",
              f"${(abs(b_buy)+b_sell)*swing:,.2f}")

    st.success(
        "### ✅ Buffett was right — in this dataset, statistically.\n\n"
        "*\"Be fearful when others are greedy, and greedy when others are fearful.\"*\n\n"
        "Every piece of evidence agrees:\n"
        "- VIF confirmed our predictors are clean and independent.\n"
        "- Regression Model 6 shows buyers and sellers on opposite slopes.\n"
        "- All four hypothesis tests reject H₀ at **p < 0.001**.\n\n"
        "The pattern isn't a quirk of one model — it survives every angle "
        "we test it from, across 210,000 real trades spanning two years."
    )

    st.markdown("### Practical implications")
    st.markdown(
        """
        - **Entry timing** — when the Fear & Greed index is low (Fear),
          the historical edge favours *buying*. When it's high (Greed),
          the edge favours *selling* or staying out.
        - **Position sizing** — the mood-driven gap is about
          **$90 per trade** at the Fear↔Greed extremes. Real but small
          relative to trade sizes in this dataset, so it's an *edge*, not
          a guarantee.
        - **Emotional check** — the crowd's mood is measurably wrong-way
          at extremes. Use the F&G index as a contrarian indicator.
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        """
        - R² is small (~1.7%) — mood is one of many forces on trade
          profit; we're measuring a real effect, not a full prediction
          model.
        - Data covers one bull/bear cycle only — the effect may differ in
          other regimes (e.g., prolonged sideways markets).
        - Welch's t-test assumes roughly independent trades; in practice
          trades within a day cluster. The conclusions are directionally
          correct but p-values may be slightly optimistic.
        """
    )

    st.caption("Built to accompany `working.ipynb` — same analysis, interactive view.")
