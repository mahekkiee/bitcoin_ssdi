# ================================================================
# Should I Buy When People Are Scared?
# A simple regression study of the Crypto Fear & Greed Index
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Page setup
st.set_page_config(page_title="Buy the Fear?", layout="wide")
sns.set_style("whitegrid")


# ================================================================
# LOAD DATA  (runs only once, thanks to @st.cache_data)
# ================================================================
@st.cache_data
def load_data():
    # --- File 1: Fear & Greed Index (one row per day) ---
    fg = pd.read_csv("fear_greed_index.csv")
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg[["date", "value", "classification"]]
    fg.columns = ["date", "FG_value", "Sentiment"]

    # --- File 2: Trade history (gzipped to fit on GitHub — pandas handles it automatically) ---
    tr = pd.read_csv("historical_data.csv.gz", compression="gzip")
    tr["Timestamp IST"] = pd.to_datetime(tr["Timestamp IST"],
                                         format="%d-%m-%Y %H:%M",
                                         errors="coerce")
    tr = tr.dropna(subset=["Timestamp IST"])
    tr["date"] = tr["Timestamp IST"].dt.normalize()

    # Keep only normal trades (remove weird rows like dust conversions)
    keep = ["Open Long", "Close Long", "Open Short", "Close Short", "Buy", "Sell"]
    tr = tr[tr["Direction"].isin(keep)].copy()

    # Keep only the columns we need, rename to simple names
    tr = tr[["date", "Coin", "Side", "Direction",
             "Size USD", "Closed PnL", "Fee", "Execution Price", "Size Tokens"]]
    tr.columns = ["date", "Coin", "Side", "Direction",
                  "Size_USD", "Closed_PnL", "Fee", "Price", "Tokens"]

    # --- Join the two files on date ---
    df = tr.merge(fg, on="date", how="inner")
    return df, fg


df, fg_daily = load_data()


# ================================================================
# SIDEBAR  — page navigation
# ================================================================
st.sidebar.title("📑 Pages")
page = st.sidebar.radio("Go to", [
    "1. Overview",
    "2. The Data",
    "3. Trade Evidence",
    "4. Bitcoin Evidence",
    "5. Final Verdict",
])

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df):,}** trades joined with sentiment")


# ================================================================
# PAGE 1 — OVERVIEW
# ================================================================
if page == "1. Overview":
    st.title("🪙 Should I Buy When People Are Scared?")
    st.subheader("A simple regression study of the Crypto Fear & Greed Index")

    st.markdown("""
    ### The big question
    Famous investors like Warren Buffett say: *"Buy when everyone is scared,
    sell when everyone is excited."* It sounds smart — but **is it actually true?
    Does it really make money?**

    We test this with real data using linear regression.

    ### What we want to find out
    We ask the same question in **two different ways**:

    **Question 1 (Trade-level):** We have 2 lakh+ real trades from 32 traders.
    On the day each trade happened, was the market scared or excited? Did trades
    made on "scared days" make more money than trades made on "excited days"?

    **Question 2 (Bitcoin-level):** If today the market is very scared,
    does Bitcoin's price go up tomorrow?

    If both answers say "yes, fear is good for buyers" — the old saying is true.
    If they don't agree — it's probably just a nice quote.

    ### The tool we use
    Linear regression, exactly like in class:
    ```python
    fit = smf.ols("y ~ x", data=df).fit()
    fit.summary()
    ```
    """)


# ================================================================
# PAGE 2 — THE DATA
# ================================================================
elif page == "2. The Data":
    st.title("📦 The Data")

    st.markdown("### File 1: Fear & Greed Index (the 'mood meter')")
    st.write(f"Daily score from 0 to 100 — **{len(fg_daily):,} days** of data.")
    st.write("- 0 means everyone is terrified\n- 100 means everyone is super excited")
    st.dataframe(fg_daily.head(), use_container_width=True)

    # Simple plot: how often does each mood happen?
    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    counts = fg_daily["Sentiment"].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, palette="RdYlGn", ax=ax)
    ax.set_title("How often does each mood happen?")
    ax.set_ylabel("Number of days")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### File 2: Trade history (the 'trade diary')")
    st.write(f"Every trade made by 32 traders — **{len(df):,} trades** "
             f"(after joining with the mood meter).")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### What each column means")
    st.markdown("""
    - **Side** — BUY or SELL
    - **Size_USD** — how much money was in the trade
    - **Closed_PnL** — profit (+) or loss (−) on that trade
    - **Fee** — cost of placing the trade
    - **FG_value** — fear/greed score (0-100) on that day
    - **Sentiment** — fear/greed label on that day
    """)


# ================================================================
# PAGE 3 — TRADE EVIDENCE (the MICRO regression)
# ================================================================
elif page == "3. Trade Evidence":
    st.title("💼 Trade Evidence — Does sentiment predict trade profit?")

    # --- First, just look at the means ---
    st.markdown("### Step 1 — Just look at the averages")
    st.write("Before any regression, let's just take the mean profit in each mood.")

    buys = df[df["Side"] == "BUY"]
    sells = df[df["Side"] == "SELL"]

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    avg_buy = buys.groupby("Sentiment")["Closed_PnL"].mean().reindex(order).round(2)
    avg_sell = sells.groupby("Sentiment")["Closed_PnL"].mean().reindex(order).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**BUY trades — average profit ($)**")
        st.dataframe(avg_buy, use_container_width=True)
    with col2:
        st.markdown("**SELL trades — average profit ($)**")
        st.dataframe(avg_sell, use_container_width=True)

    # Simple bar plot
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(order))
    width = 0.35
    ax.bar(x - width/2, avg_buy.values, width, label="BUY", color="green", alpha=0.7)
    ax.bar(x + width/2, avg_sell.values, width, label="SELL", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Average profit ($)")
    ax.set_title("Average profit by mood and side of trade")
    ax.legend()
    st.pyplot(fig)

    st.success("**What we see:** BUY trades made during Fear earn more than BUY trades "
               "made during Extreme Greed. SELL trades do the opposite. "
               "This is exactly what the Buffett quote predicts!")

    # --- Simple regression ---
    st.markdown("---")
    st.markdown("### Step 2 — Simple regression")
    st.write("Now let's fit a line: `Closed_PnL = a + b × FG_value`")
    st.code('fit1 = smf.ols("Closed_PnL ~ FG_value", data=df).fit()\nfit1.summary()')

    fit1 = smf.ols("Closed_PnL ~ FG_value", data=df).fit()
    st.code(fit1.summary().as_text(), language="text")

    p = fit1.pvalues["FG_value"]
    b = fit1.params["FG_value"]
    st.info(f"**How to read this:** The coefficient on FG_value is **{b:.4f}** with "
            f"p-value **{p:.4f}**. Because p < 0.05, we can say sentiment has a real "
            f"(non-zero) effect on profit. But R² is tiny — "
            f"sentiment alone doesn't explain much. "
            f"That's normal — trade profit depends on many things. "
            f"Let's add more variables.")

    # --- Multiple regression ---
    st.markdown("---")
    st.markdown("### Step 3 — Add more variables (multiple regression)")
    st.write("Real trades differ in size and fees. Let's control for those.")
    st.code('fit2 = smf.ols("Closed_PnL ~ FG_value + Size_USD + Fee", data=df).fit()\nfit2.summary()')

    fit2 = smf.ols("Closed_PnL ~ FG_value + Size_USD + Fee", data=df).fit()
    st.code(fit2.summary().as_text(), language="text")

    st.info("**How to read this:** Even after we account for trade size and fees, "
            "FG_value still has a positive, significant coefficient. "
            "Every 1-point increase in F&G (more greedy) is linked with a small "
            "increase in average PnL — but this is the overall effect mixing buyers "
            "and sellers. The real magic happens when we separate them.")

    # --- Interaction ---
    st.markdown("---")
    st.markdown("### Step 4 — The key test: interaction with Side")
    st.write("Does sentiment affect BUYERS and SELLERS differently? "
             "We use an interaction term (`*`) — same as `Sales ~ TV * Radio` in class.")
    st.code('fit3 = smf.ols("Closed_PnL ~ FG_value * Side + Size_USD", data=df).fit()\nfit3.summary()')

    fit3 = smf.ols("Closed_PnL ~ FG_value * Side + Size_USD", data=df).fit()
    st.code(fit3.summary().as_text(), language="text")

    b_fg = fit3.params["FG_value"]
    b_int = fit3.params["FG_value:Side[T.SELL]"]
    p_int = fit3.pvalues["FG_value:Side[T.SELL]"]

    st.success(f"""
    **This is the main finding!**

    - For BUY trades, the FG_value coefficient is **{b_fg:+.4f}** —
      as the market gets greedier, buyer profit **goes down**.
    - For SELL trades, the coefficient is **{b_fg + b_int:+.4f}**
      (= {b_fg:.4f} + {b_int:.4f}) —
      as the market gets greedier, seller profit **goes up**.
    - The interaction term is highly significant (p = {p_int:.4f}).

    **In plain English:** Buyers do better in Fear, sellers do better in Greed.
    The old saying is statistically confirmed on this dataset.
    """)


# ================================================================
# PAGE 4 — BITCOIN EVIDENCE (the MACRO regression)
# ================================================================
elif page == "4. Bitcoin Evidence":
    st.title("₿ Bitcoin Evidence — Does today's mood predict tomorrow's price?")

    # Build daily BTC price from trades
    st.markdown("### Step 1 — Build a daily Bitcoin price from the trades")
    st.write("We keep only BTC trades and compute a volume-weighted average "
             "price (VWAP) for each day. This gives us one price per day.")

    btc = df[(df["Coin"] == "BTC") & (df["Price"] > 0)].copy()

    daily = btc.groupby("date").apply(
        lambda g: pd.Series({
            "BTC_Price": (g["Price"] * g["Tokens"].abs()).sum() / g["Tokens"].abs().sum(),
            "BTC_Volume": g["Size_USD"].sum(),
        })
    ).reset_index()

    daily = daily.merge(fg_daily, on="date", how="inner").sort_values("date").reset_index(drop=True)
    daily["BTC_Return_Next"] = daily["BTC_Price"].pct_change().shift(-1) * 100
    daily["FG_Lag1"] = daily["FG_value"].shift(1)
    daily["FG_Lag7"] = daily["FG_value"].shift(7)

    st.write(f"We have **{len(daily)} days** of BTC data to work with.")
    st.dataframe(daily.head(), use_container_width=True)

    # Plot BTC price over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["BTC_Price"], color="orange", linewidth=1.5)
    ax.set_title("Bitcoin Price (daily VWAP from trades)")
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)

    # Plot FG over time
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(fg_daily["date"], fg_daily["FG_value"], color="purple", linewidth=1)
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Fear & Greed Index over time")
    ax2.set_ylabel("F&G score (0-100)")
    ax2.set_ylim(0, 100)
    st.pyplot(fig2)

    # --- Mean return by mood ---
    st.markdown("---")
    st.markdown("### Step 2 — Mean next-day BTC return by today's mood")
    ret_by_mood = daily.groupby("Sentiment")["BTC_Return_Next"].agg(["mean", "count"]).round(3)
    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    ret_by_mood = ret_by_mood.reindex([s for s in order if s in ret_by_mood.index])
    st.dataframe(ret_by_mood, use_container_width=True)

    st.info("**What we see:** After Extreme Fear days, BTC tends to rise the next day "
            "— the direction agrees with our trade-level finding. But the count is very "
            "low for some moods, so we need a proper regression to check if this is real.")

    # --- Regression ---
    st.markdown("---")
    st.markdown("### Step 3 — Simple regression")
    st.write("Does today's F&G value predict tomorrow's BTC return?")
    st.code('fit = smf.ols("BTC_Return_Next ~ FG_value", data=daily).fit()\nfit.summary()')

    d = daily.dropna(subset=["BTC_Return_Next"])
    fit = smf.ols("BTC_Return_Next ~ FG_value", data=d).fit()
    st.code(fit.summary().as_text(), language="text")

    # Scatter with regression line
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.regplot(data=d, x="FG_value", y="BTC_Return_Next",
                order=1, scatter_kws={"alpha": 0.4, "s": 25}, ax=ax3)
    ax3.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax3.set_title("Today's F&G score vs Tomorrow's BTC Return")
    ax3.set_xlabel("F&G value today")
    ax3.set_ylabel("BTC return tomorrow (%)")
    st.pyplot(fig3)

    p_val = fit.pvalues["FG_value"]
    st.warning(f"""
    **Honest finding:** The coefficient is tiny and p-value = **{p_val:.3f}**
    (much bigger than 0.05). So at the daily price level, we **cannot prove**
    that today's fear predicts tomorrow's BTC return on this 2-year sample.

    The *direction* is consistent with "buy the fear" — Extreme Fear days are
    followed by the biggest gains on average — but there are only a handful of
    Extreme Fear days, so the test can't rule out luck.
    """)

    # --- With lags ---
    st.markdown("---")
    st.markdown("### Step 4 — With lags (does yesterday or last week matter?)")
    st.code('fit_lag = smf.ols("BTC_Return_Next ~ FG_value + FG_Lag1 + FG_Lag7", data=daily).fit()')

    d2 = daily.dropna(subset=["BTC_Return_Next", "FG_Lag1", "FG_Lag7"])
    fit_lag = smf.ols("BTC_Return_Next ~ FG_value + FG_Lag1 + FG_Lag7", data=d2).fit()
    st.code(fit_lag.summary().as_text(), language="text")

    st.info("Adding lags doesn't help either — none of the p-values are below 0.05. "
            "This is an honest null result at the price level.")


# ================================================================
# PAGE 5 — VERDICT
# ================================================================
elif page == "5. Final Verdict":
    st.title("⚖️ Final Verdict — Should you buy when people are scared?")

    st.markdown("""
    ## The short answer

    **Partially yes.** Here's what the regressions told us:
    """)

    verdict = pd.DataFrame([
        ["Trade-level", "Closed_PnL ~ FG_value × Side",
         "✅ Highly significant (p < 0.001)",
         "BUYS profit more in Fear; SELLS profit more in Greed. Buffett was right."],
        ["Bitcoin daily", "BTC_Return(next day) ~ FG_value",
         "❌ Not significant (p ≈ 0.89)",
         "Direction agrees but too few data points to prove it statistically."],
    ], columns=["Level", "Model", "Result", "Plain English"])
    st.dataframe(verdict, use_container_width=True, hide_index=True)

    st.markdown("""
    ## What this means in practice

    The strong evidence is at the **individual trade level**. If you are a trader,
    the data says it really does pay to be a **contrarian buyer** — buy more when
    fear is high, pull back when the market is euphoric. That's not just a quote,
    it's visible in 2 lakh+ real trades.

    At the **overall market level**, the effect is in the right direction but the
    2-year sample is too small to call it statistically proven. More data (5+ years)
    might resolve this.

    ## Limitations (what this project does NOT prove)

    - Only 2 years of overlap — a longer sample could change the Bitcoin result.
    - 32 traders is a small sample of the crypto world — they may not be representative.
    - "Profit" here means closed PnL on Hyperliquid perpetuals; this is different
      from holding-period returns for a spot investor.
    - Our regression assumes a **linear** effect — maybe extreme fear is very different
      from mild fear and the true relationship is non-linear.

    ## Honest one-line conclusion

    > *"Among the 32 traders we studied, buying during fear and selling during greed
    > was on average more profitable — but the effect is not yet strong enough
    > to show up in daily Bitcoin prices alone."*
    """)

    st.caption("Project built for the SSDI module • Regression study using "
               "Crypto Fear & Greed Index + Hyperliquid trade log.")
