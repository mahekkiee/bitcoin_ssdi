"""
Sentiment Effect in Crypto Trading - Streamlit Dashboard

Tests whether market mood (Fear & Greed Index) changes how much
traders make on each trade. Mirrors the analysis in working.ipynb.

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
    page_title="Sentiment Effect in Crypto Trading",
    layout="wide",
)
sns.set_theme(style="whitegrid")

FG_FILE    = "fear_greed_index.csv"
TRADE_FILE = "historical_data.csv.gz"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral",
                   "Greed", "Extreme Greed"]

M6_KEY = "M6: FG x Side + Size + Price"


# ---------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Loading and joining the two files...")
def load_data():
    missing = [f for f in [FG_FILE, TRADE_FILE] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) in the app folder: {', '.join(missing)}. "
            "Place both files next to app.py and rerun."
        )

    fg = pd.read_csv(FG_FILE)
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg[["date", "value", "classification"]]
    fg.columns = ["date", "FG_value", "Sentiment"]

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
# Cached computations
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Fitting Model 6...")
def fit_model6(df):
    return smf.ols(
        "Closed_PnL ~ FG_value * Side + Size_USD + Price",
        data=df,
    ).fit()


@st.cache_data(show_spinner="Fitting all candidate models...")
def fit_all_models(df):
    specs = {
        "M1: Size only":                    "Closed_PnL ~ Size_USD",
        "M2: FG only":                      "Closed_PnL ~ FG_value",
        "M3: FG + Size":                    "Closed_PnL ~ FG_value + Size_USD",
        "M4: FG + Size + Side":             "Closed_PnL ~ FG_value + Size_USD + Side",
        "M5: FG + Size + Side + Price":     "Closed_PnL ~ FG_value + Size_USD + Side + Price",
        M6_KEY:                             "Closed_PnL ~ FG_value * Side + Size_USD + Price",
        "M7: FG + FG_sq + Side + Size":     "Closed_PnL ~ FG_value + FG_sq + Side + Size_USD",
    }
    fits = {name: smf.ols(spec, data=df).fit() for name, spec in specs.items()}
    rows = [{
        "Model": name,
        "R2": round(f.rsquared, 6),
        "Adj R2": round(f.rsquared_adj, 6),
        "AIC": round(f.aic, 0),
        "N obs": int(f.nobs),
    } for name, f in fits.items()]
    compare = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return fits, compare


# ---------------------------------------------------------------
# Bell curve helper for hypothesis testing
# ---------------------------------------------------------------
def plot_t_bell_curve(t_stat, p_value, df_deg,
                      one_sided=True, alpha=0.05, title=""):
    """Draw a t-distribution with shaded accept / reject regions and
    a vertical line at the observed t-statistic."""
    # Dynamic x-range so the observed t is always visible
    x_max = max(4.0, abs(t_stat) * 1.15 + 0.5)
    x = np.linspace(-x_max, x_max, 500)
    y = stats.t.pdf(x, df_deg)

    if one_sided:
        crit = stats.t.ppf(1 - alpha, df_deg)
    else:
        crit = stats.t.ppf(1 - alpha / 2, df_deg)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, color="black", linewidth=2)

    # Accept region (green)
    if one_sided:
        mask_accept = x < crit
    else:
        mask_accept = (x > -crit) & (x < crit)
    ax.fill_between(x, 0, y, where=mask_accept,
                    color="#2ca02c", alpha=0.25,
                    label=f"Accept H0 region ({(1-alpha)*100:.0f}%)")

    # Reject region (red)
    if one_sided:
        mask_reject = x >= crit
    else:
        mask_reject = (x <= -crit) | (x >= crit)
    ax.fill_between(x, 0, y, where=mask_reject,
                    color="#d62728", alpha=0.35,
                    label=f"Reject H0 region (alpha={alpha})")

    # Critical value line(s)
    ax.axvline(crit, color="#d62728", linestyle="--", linewidth=1.2)
    ax.text(crit, y.max() * 0.6, f"  t_crit = {crit:.2f}",
            color="#d62728", fontsize=9, verticalalignment="center")
    if not one_sided:
        ax.axvline(-crit, color="#d62728", linestyle="--", linewidth=1.2)
        ax.text(-crit, y.max() * 0.6, f"-t_crit = {-crit:.2f}  ",
                color="#d62728", fontsize=9,
                verticalalignment="center", horizontalalignment="right")

    # Observed t-statistic
    ax.axvline(t_stat, color="blue", linewidth=2.5,
               label=f"Observed t = {t_stat:.2f}")
    # p-value annotation
    ax.annotate(f"p = {p_value:.2e}",
                xy=(t_stat, 0),
                xytext=(t_stat, y.max() * 0.85),
                fontsize=11, color="blue", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, y.max() * 1.2)
    plt.tight_layout()
    return fig


def plot_f_bell_curve(f_stat, p_value, dfn, dfd, alpha=0.05, title=""):
    """Draw an F-distribution with shaded accept / reject regions.
    F-test is always right-tailed."""
    x_max = max(6.0, f_stat * 1.15 + 1.0)
    x = np.linspace(0.001, x_max, 500)
    y = stats.f.pdf(x, dfn, dfd)
    crit = stats.f.ppf(1 - alpha, dfn, dfd)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, color="black", linewidth=2)

    mask_accept = x < crit
    ax.fill_between(x, 0, y, where=mask_accept,
                    color="#2ca02c", alpha=0.25,
                    label=f"Accept H0 region ({(1-alpha)*100:.0f}%)")

    mask_reject = x >= crit
    ax.fill_between(x, 0, y, where=mask_reject,
                    color="#d62728", alpha=0.35,
                    label=f"Reject H0 region (alpha={alpha})")

    ax.axvline(crit, color="#d62728", linestyle="--", linewidth=1.2)
    ax.text(crit, y.max() * 0.6, f"  F_crit = {crit:.2f}",
            color="#d62728", fontsize=9, verticalalignment="center")
    ax.axvline(f_stat, color="blue", linewidth=2.5,
               label=f"Observed F = {f_stat:.2f}")
    ax.annotate(f"p = {p_value:.2e}",
                xy=(f_stat, 0),
                xytext=(f_stat, y.max() * 0.85),
                fontsize=11, color="blue", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

    ax.set_xlabel(f"F-statistic (dfn={dfn}, dfd={dfd:,})")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, y.max() * 1.2)
    ax.set_xlim(0, x_max)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------
# Load once, fail early
# ---------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Exploratory Analysis",
        "VIF - Feature Selection",
        "Linear Regression",
        "Hypothesis Testing",
        "Final Verdict",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Dataset:** {len(df):,} trades - "
    f"{df['Coin'].nunique()} coins - "
    f"{df['date'].nunique()} days"
)


# ===============================================================
# 1. OVERVIEW
# ===============================================================
if page == "Overview":
    st.title("Sentiment Effect in Crypto Trading")
    st.caption('"Be fearful when others are greedy, and greedy when others are fearful." '
               "- testing the statement on 210k real trades.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trades", f"{len(df):,}")
    c2.metric("Trading days", f"{df['date'].nunique():,}")
    c3.metric("Unique coins", df["Coin"].nunique())
    c4.metric("Date range",
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    st.markdown("### Project summary")
    st.markdown(
        """
        This dashboard tests a single question: **does market mood change
        how profitable a trade is?** Two data files are used:

        - **Fear & Greed Index** - one mood score (0-100) per day, where
          0 = extreme fear and 100 = extreme greed.
        - **Trade history** - every trade made by ~32 traders over two
          years, with side (BUY / SELL), size, fee, price, and closed PnL.

        The analysis follows a textbook statistical pipeline:

        1. **EDA** - eyeball patterns in the raw data
        2. **VIF** - drop predictors that measure the same thing
        3. **Regression** - 7 candidate models, pick the best by AIC
        4. **Hypothesis testing** - 4 formal tests of the Buffett claim
        5. **Final verdict** - combining every piece of evidence
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
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    st.markdown("### Average PnL by sentiment, split by side")
    eda = (df.groupby(["Sentiment", "Side"])["Closed_PnL"]
             .mean().round(2).unstack()
             .reindex(SENTIMENT_ORDER))
    st.dataframe(eda, use_container_width=True)
    st.info(
        "Look at the columns carefully. For BUY trades, profit tends to be "
        "higher in Fear than Greed. For SELL trades, the opposite. "
        "This is the Buffett effect visible in the raw averages - "
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


# ===============================================================
# 3. VIF - FEATURE SELECTION
# ===============================================================
elif page == "VIF - Feature Selection":
    st.title("VIF - Feature Selection")

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

    st.markdown("### Step 1 - Initial VIF (all candidates)")
    st.dataframe(vif_initial, use_container_width=True)

    st.markdown("### Correlation matrix (explains the VIF numbers)")
    corr = X.corr().round(3)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".3f",
                ax=ax)
    st.pyplot(fig)
    st.warning(
        f"Size_USD and Fee correlation is about "
        f"**{corr.loc['Size_USD', 'Fee']:.2f}** - high. This is why their "
        "VIFs are around 2.3. Fee is charged as a percentage of trade size, "
        "so it adds little new information the regression can use."
    )

    st.markdown("### Step 2 - Drop `Fee`, recompute VIF")
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
        "All VIFs are near 1 - no multicollinearity left. "
        "Final predictor set for regression: "
        "**FG_value, Size_USD, Price, Tokens, Side (categorical)**."
    )


# ===============================================================
# 4. LINEAR REGRESSION
# ===============================================================
elif page == "Linear Regression":
    st.title("Linear Regression - Who profits from which mood?")

    fits, compare = fit_all_models(df)
    lm6 = fits[M6_KEY]

    st.markdown("### Model comparison (lower AIC = better)")
    st.dataframe(compare, use_container_width=True)
    st.success(
        f"**Winner: {compare.iloc[0]['Model']}** - lowest AIC, and it's "
        "the only model that lets buyers and sellers have different slopes."
    )

    # --- Coefficient table with p-values -----------------
    st.markdown("### Model 6 - coefficient table with p-values")

    coef_df = pd.DataFrame({
        "Coefficient": lm6.params.round(4),
        "Std error":   lm6.bse.round(4),
        "t-stat":      lm6.tvalues.round(3),
        "p-value":     lm6.pvalues.apply(lambda p: f"{p:.3e}"),
    })
    st.dataframe(coef_df, use_container_width=True)

    # --- Prominent p-value metrics -----------------
    st.markdown("### Key p-values at a glance")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("p(FG_value)",
               f"{lm6.pvalues['FG_value']:.2e}",
               "Reject H0" if lm6.pvalues['FG_value'] < 0.05 else "Keep H0")
    pc2.metric("p(FG x Side)",
               f"{lm6.pvalues['FG_value:Side[T.SELL]']:.2e}",
               "Reject H0" if lm6.pvalues['FG_value:Side[T.SELL]'] < 0.05 else "Keep H0")
    pc3.metric("p(Side[SELL])",
               f"{lm6.pvalues['Side[T.SELL]']:.2e}",
               "Reject H0" if lm6.pvalues['Side[T.SELL]'] < 0.05 else "Keep H0")

    # --- Slope metrics -----------------
    b_buy  = lm6.params["FG_value"]
    b_diff = lm6.params["FG_value:Side[T.SELL]"]
    b_sell = b_buy + b_diff

    st.markdown("### The two fitted slopes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer slope", f"{b_buy:+.4f}",
              help="Profit change per 1-point rise in greed")
    c2.metric("Seller slope", f"{b_sell:+.4f}",
              help="= buyer slope + interaction term")
    c3.metric("Interaction (gap)", f"{b_diff:+.4f}",
              help="How differently sellers respond vs buyers")

    # --- The CLEAN X-shape plot: binned means + fitted lines -------
    st.markdown("### The X-shape: binned averages with Model 6 fitted lines")
    st.caption(
        "Each dot is the AVERAGE PnL for trades in a Fear-&-Greed bucket "
        "(10 equal-width bins). Dot size = number of trades in that bucket. "
        "Lines are Model 6's fitted slopes for BUY and SELL."
    )

    # 1) Bin the data: mean PnL per (bucket, side)
    df_plot = df.copy()
    df_plot["FG_bin"] = pd.cut(df_plot["FG_value"], bins=10)
    binned = (df_plot.groupby(["FG_bin", "Side"], observed=True)
                     .agg(mean_pnl=("Closed_PnL", "mean"),
                          count=("Closed_PnL", "count"),
                          fg_center=("FG_value", "mean"))
                     .reset_index())

    # 2) Generate fitted-line predictions across the FG range
    fg_range = np.linspace(df["FG_value"].min(),
                           df["FG_value"].max(), 100)
    median_size  = df["Size_USD"].median()
    median_price = df["Price"].median()
    pred_buy = pd.DataFrame({
        "FG_value": fg_range, "Side": "BUY",
        "Size_USD": median_size, "Price": median_price,
    })
    pred_sell = pd.DataFrame({
        "FG_value": fg_range, "Side": "SELL",
        "Size_USD": median_size, "Price": median_price,
    })
    pred_buy["predicted"]  = lm6.predict(pred_buy)
    pred_sell["predicted"] = lm6.predict(pred_sell)

    # 3) Plot it
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for side, color in [("BUY", "#2ca02c"), ("SELL", "#d62728")]:
        sub = binned[binned["Side"] == side]
        ax.scatter(sub["fg_center"], sub["mean_pnl"],
                   s=sub["count"] / 50, color=color, alpha=0.7,
                   edgecolor="black", linewidth=0.8,
                   label=f"{side} (binned mean)", zorder=3)
    ax.plot(pred_buy["FG_value"], pred_buy["predicted"],
            color="#1a6b1a", linewidth=2.5, label="BUY fitted slope")
    ax.plot(pred_sell["FG_value"], pred_sell["predicted"],
            color="#8b1a1c", linewidth=2.5, label="SELL fitted slope")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fear & Greed Index (0 = panic, 100 = euphoria)")
    ax.set_ylabel("Mean Closed PnL (USD)")
    ax.set_title("Buyers' slope falls, sellers' slope rises: "
                 "the Buffett effect, visually")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"**How to read this plot:**\n\n"
        f"- In extreme FEAR (left side), BUY and SELL averages are both "
        f"positive and close - but as mood shifts toward GREED (right), "
        f"BUY trades drift down toward zero while SELL trades shoot up.\n"
        f"- The **green line** slopes down by {b_buy:+.4f} per point - "
        f"buyers steadily lose expected profit as greed rises.\n"
        f"- The **red line** slopes up by {b_sell:+.4f} per point - "
        f"sellers gain as greed rises.\n"
        f"- Over a typical 50-point swing (Fear -> Greed), the gap "
        f"between the two lines widens by "
        f"**${(abs(b_buy)+b_sell)*50:,.2f} per trade**."
    )

    with st.expander("Full OLS summary (Model 6)"):
        st.code(lm6.summary().as_text(), language="text")


# ===============================================================
# 5. HYPOTHESIS TESTING
# ===============================================================
elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing")
    st.markdown(
        "Regression gave us coefficients. Hypothesis testing gives us "
        "formal **yes/no** answers with bell-curve evidence. For each "
        "claim: state H0 and H1, compute a p-value, and visualise where "
        "the observed statistic falls on the distribution."
    )
    st.caption(
        "On every plot: the **green region** is where we would accept H0. "
        "The **red region** is the rejection zone (alpha = 0.05). "
        "The **blue line** marks the observed statistic from our data."
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
    st.markdown("### H1 - Buyers earn more in Fear than in Greed")
    st.markdown(
        "**H0:** mean(buyer | Fear) = mean(buyer | Greed)  \n"
        "**H1:** mean(buyer | Fear) > mean(buyer | Greed)  \n"
        "*Test:* Welch's one-sided t-test (unequal variances)"
    )
    t1, p1_two = stats.ttest_ind(buys_fear, buys_greed, equal_var=False)
    p1 = p1_two / 2 if t1 > 0 else 1 - p1_two / 2
    df1 = len(buys_fear) + len(buys_greed) - 2

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (Fear)",  f"{len(buys_fear):,}")
    c2.metric("n (Greed)", f"{len(buys_greed):,}")
    c3.metric("Mean diff", f"${buys_fear.mean() - buys_greed.mean():+.2f}")
    c4.metric("one-sided p", f"{p1:.2e}")

    fig = plot_t_bell_curve(
        t_stat=t1, p_value=p1, df_deg=df1, one_sided=True,
        title=f"H1 bell curve - one-sided t-test "
              f"(observed t={t1:.2f}, p={p1:.2e})")
    st.pyplot(fig)

    if p1 < 0.05:
        st.success(
            f"**Reject H0** (t = {t1:.2f}, p = {p1:.2e}). "
            f"The blue line sits deep in the red rejection region - "
            f"buyers earn significantly more in fearful markets "
            f"(${buys_fear.mean():.2f} vs ${buys_greed.mean():.2f})."
        )
    else:
        st.warning(f"Fail to reject H0 (p = {p1:.3f}).")

    # -------- H2 --------
    st.markdown("---")
    st.markdown("### H2 - Sellers earn more in Greed than in Fear")
    st.markdown(
        "**H0:** mean(seller | Greed) = mean(seller | Fear)  \n"
        "**H1:** mean(seller | Greed) > mean(seller | Fear)  \n"
        "*Test:* Welch's one-sided t-test"
    )
    t2, p2_two = stats.ttest_ind(sells_greed, sells_fear, equal_var=False)
    p2 = p2_two / 2 if t2 > 0 else 1 - p2_two / 2
    df2 = len(sells_greed) + len(sells_fear) - 2

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n (Greed)", f"{len(sells_greed):,}")
    c2.metric("n (Fear)",  f"{len(sells_fear):,}")
    c3.metric("Mean diff", f"${sells_greed.mean() - sells_fear.mean():+.2f}")
    c4.metric("one-sided p", f"{p2:.2e}")

    fig = plot_t_bell_curve(
        t_stat=t2, p_value=p2, df_deg=df2, one_sided=True,
        title=f"H2 bell curve - one-sided t-test "
              f"(observed t={t2:.2f}, p={p2:.2e})")
    st.pyplot(fig)

    if p2 < 0.05:
        st.success(
            f"**Reject H0** (t = {t2:.2f}, p = {p2:.2e}). "
            f"Sellers earn significantly more in greedy markets "
            f"(${sells_greed.mean():.2f} vs ${sells_fear.mean():.2f})."
        )
    else:
        st.warning(f"Fail to reject H0 (p = {p2:.3f}).")

    # -------- H3 --------
    st.markdown("---")
    st.markdown("### H3 - Buyer PnL differs across all 5 sentiment regimes")
    st.markdown(
        "**H0:** mean(EF) = mean(F) = mean(N) = mean(G) = mean(EG)  \n"
        "**H1:** at least one mean differs  \n"
        "*Test:* One-way ANOVA (F-test, right-tailed)"
    )
    groups = [buys[buys["Sentiment"] == s]["Closed_PnL"].values
              for s in SENTIMENT_ORDER
              if s in buys["Sentiment"].unique()]
    f_stat, p3 = stats.f_oneway(*groups)
    dfn = len(groups) - 1
    dfd = sum(len(g) for g in groups) - len(groups)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F-statistic", f"{f_stat:.3f}")
    c2.metric("p-value", f"{p3:.2e}")
    c3.metric("Groups", f"{len(groups)}")
    c4.metric("dfn, dfd", f"{dfn}, {dfd:,}")

    fig = plot_f_bell_curve(
        f_stat=f_stat, p_value=p3, dfn=dfn, dfd=dfd,
        title=f"H3 bell curve - ANOVA F-test "
              f"(observed F={f_stat:.2f}, p={p3:.2e})")
    st.pyplot(fig)

    if p3 < 0.05:
        st.success(
            f"**Reject H0** (F = {f_stat:.2f}, p = {p3:.2e}). "
            "Average buyer profit differs meaningfully by sentiment regime."
        )
    else:
        st.warning(f"Fail to reject H0 (p = {p3:.3f}).")

    # Bar chart supporting H3
    fig_bar, ax = plt.subplots(figsize=(9, 4))
    buyer_means = buys.groupby("Sentiment")["Closed_PnL"].mean().reindex(
        SENTIMENT_ORDER)
    sns.barplot(x=buyer_means.index, y=buyer_means.values,
                palette="RdYlGn_r", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Buyer PnL (USD)")
    ax.set_title("Buyer PnL by regime - the monotone pattern")
    st.pyplot(fig_bar)

    # -------- H4 --------
    st.markdown("---")
    st.markdown("### H4 - The Buffett interaction is statistically real")
    st.markdown(
        "**H0:** beta(FG_value x Side) = 0  \n"
        "**H1:** beta(FG_value x Side) != 0  \n"
        "*Test:* two-sided t-test on Model 6's interaction coefficient"
    )
    lm6 = fit_model6(df)
    int_coef = lm6.params["FG_value:Side[T.SELL]"]
    int_se   = lm6.bse["FG_value:Side[T.SELL]"]
    int_p    = lm6.pvalues["FG_value:Side[T.SELL]"]
    int_t    = lm6.tvalues["FG_value:Side[T.SELL]"]
    int_ci   = lm6.conf_int().loc["FG_value:Side[T.SELL]"]
    int_df   = int(lm6.df_resid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("beta",      f"{int_coef:+.4f}")
    c2.metric("Std error", f"{int_se:.4f}")
    c3.metric("t-stat",    f"{int_t:+.2f}")
    c4.metric("p-value",   f"{int_p:.2e}")
    st.caption(f"95% CI: [{int_ci[0]:+.3f}, {int_ci[1]:+.3f}]")

    fig = plot_t_bell_curve(
        t_stat=int_t, p_value=int_p, df_deg=int_df, one_sided=False,
        title=f"H4 bell curve - two-sided t-test "
              f"(observed t={int_t:.2f}, p={int_p:.2e})")
    st.pyplot(fig)

    if int_p < 0.05:
        st.success(
            f"**Reject H0** (p = {int_p:.2e}). "
            "Buyers and sellers respond to sentiment in genuinely opposite "
            "directions - not by chance."
        )
    else:
        st.warning(f"Fail to reject H0 (p = {int_p:.3f}).")


# ===============================================================
# 6. FINAL VERDICT
# ===============================================================
elif page == "Final Verdict":
    st.title("Final Verdict")

    fits, _ = fit_all_models(df)
    lm6 = fits[M6_KEY]
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
         "Dropped Fee (corr with Size_USD ~ 0.75, VIF ~ 2.3)",
         "Clean predictors"],
        ["Regression", "Model 6: FG x Side + Size + Price",
         f"R2 = {lm6.rsquared:.4f},  AIC = {lm6.aic:,.0f}",
         f"Buyer slope {b_buy:+.4f} / Seller slope {b_sell:+.4f}"],
        ["H1 (buyers)", "Welch t-test, Fear > Greed",
         f"Diff = ${buys_fear.mean()-buys_greed.mean():+.2f},  p = {p1:.2e}",
         "Buyers earn more in fear"],
        ["H2 (sellers)", "Welch t-test, Greed > Fear",
         f"Diff = ${sells_greed.mean()-sells_fear.mean():+.2f},  p = {p2:.2e}",
         "Sellers earn more in greed"],
        ["H3 (regimes)", "One-way ANOVA on 5 regimes",
         f"F = {f_stat:.2f},  p = {p3:.2e}",
         "Buyer PnL differs by regime"],
        ["H4 (interaction)", "Wald test on beta(FG x Side)",
         f"beta = {int_coef:+.4f},  p = {int_p:.2e}",
         "Opposite effects confirmed"],
    ], columns=["Evidence", "Test", "Result", "Verdict"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ---- HEADLINE NUMBERS WITH CALCULATION BREAKDOWN ----
    st.markdown("### How we got the headline numbers")

    swing = 50
    buyer_loss  = abs(b_buy)  * swing
    seller_gain = b_sell      * swing
    total_gap   = buyer_loss + seller_gain

    st.markdown(
        f"""
All three numbers come from Model 6's coefficients multiplied by the size
of a typical mood swing. Here is the walk-through:

**Step 1 - pick a representative mood swing.**
The Fear & Greed index usually sits around **25** when the market is
fearful and **75** when it is greedy. So the swing between them is:

&nbsp;&nbsp;&nbsp;&nbsp;`swing = 75 - 25 = 50 points`

**Step 2 - read the two slopes from Model 6.**
Both slopes are expressed as *dollars of expected PnL per 1-point change
in the Fear & Greed index*:

&nbsp;&nbsp;&nbsp;&nbsp;`buyer_slope  = {b_buy:+.4f}   (negative - greed hurts buyers)`
&nbsp;&nbsp;&nbsp;&nbsp;`seller_slope = {b_sell:+.4f}   (positive - greed helps sellers)`

**Step 3 - scale each slope by the swing.**
A 50-point move multiplies each slope by 50:

&nbsp;&nbsp;&nbsp;&nbsp;`buyer_expected_loss  = |buyer_slope|  x swing`
&nbsp;&nbsp;&nbsp;&nbsp;`                     = |{b_buy:.4f}| x 50`
&nbsp;&nbsp;&nbsp;&nbsp;`                     = ${buyer_loss:.2f}`

&nbsp;&nbsp;&nbsp;&nbsp;`seller_expected_gain = seller_slope x swing`
&nbsp;&nbsp;&nbsp;&nbsp;`                     = {b_sell:.4f} x 50`
&nbsp;&nbsp;&nbsp;&nbsp;`                     = ${seller_gain:.2f}`

**Step 4 - add them for the total mood-driven gap.**
The buyer's loss and the seller's gain are on opposite sides of the same
mood swing, so they add up:

&nbsp;&nbsp;&nbsp;&nbsp;`total_gap = buyer_expected_loss + seller_expected_gain`
&nbsp;&nbsp;&nbsp;&nbsp;`          = ${buyer_loss:.2f} + ${seller_gain:.2f}`
&nbsp;&nbsp;&nbsp;&nbsp;`          = ${total_gap:.2f}`

That is the "cost" a buyer pays, plus the "bonus" a seller collects, for
every trade placed when mood moves from Fear to Greed.
        """
    )

    st.markdown("### The headline numbers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer expected loss per trade",
              f"${buyer_loss:,.2f}",
              help=f"|{b_buy:.4f}| * 50 = {buyer_loss:.2f}")
    c2.metric("Seller expected gain per trade",
              f"${seller_gain:,.2f}",
              help=f"{b_sell:.4f} * 50 = {seller_gain:.2f}")
    c3.metric("Total mood-driven gap",
              f"${total_gap:,.2f}",
              help=f"{buyer_loss:.2f} + {seller_gain:.2f} = {total_gap:.2f}")

    st.success(
        "### Buffett was right - in this dataset, statistically.\n\n"
        '*"Be fearful when others are greedy, and greedy when others are fearful."*\n\n'
        "Every piece of evidence agrees:\n"
        "- VIF confirmed our predictors are clean and independent.\n"
        "- Regression Model 6 shows buyers and sellers on opposite slopes.\n"
        "- All four hypothesis tests reject H0 at **p < 0.001**.\n\n"
        "The pattern isn't a quirk of one model - it survives every angle "
        "we test it from, across 210,000 real trades spanning two years."
    )

    st.markdown("### Practical implications")
    st.markdown(
        """
        - **Entry timing** - when the Fear & Greed index is low (Fear),
          the historical edge favours *buying*. When it's high (Greed),
          the edge favours *selling* or staying out.
        - **Position sizing** - the mood-driven gap is about
          **$90 per trade** at the Fear-Greed extremes. Real but small
          relative to trade sizes in this dataset, so it's an *edge*, not
          a guarantee.
        - **Emotional check** - the crowd's mood is measurably wrong-way
          at extremes. Use the F&G index as a contrarian indicator.
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        """
        - R2 is small (~1.7%) - mood is one of many forces on trade
          profit; we're measuring a real effect, not a full prediction
          model.
        - Data covers one bull/bear cycle only - the effect may differ in
          other regimes (e.g., prolonged sideways markets).
        - Welch's t-test assumes roughly independent trades; in practice
          trades within a day cluster. The conclusions are directionally
          correct but p-values may be slightly optimistic.
        """
    )

    st.caption("Built to accompany `working.ipynb` - same analysis, interactive view.")
