# 🪙 Should I Buy When People Are Scared?

A simple regression project that tests the famous investing advice
*"be greedy when others are fearful"* using real crypto data.

## What's in this repo

| File | What it is |
|---|---|
| `app.py` | The Streamlit dashboard (5 pages) |
| `requirements.txt` | Python packages needed |
| `fear_greed_index.csv` | Daily crypto Fear & Greed Index |
| `historical_data.csv` | 2 lakh+ real trades from 32 traders |
| `README.md` | This file |

## How to deploy (3 steps, ~5 minutes)

### 1. Put all 5 files in a GitHub repo
Just drag-and-drop them into a new repo at github.com. That's it.

### 2. Deploy on Streamlit Community Cloud (free)
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Pick your repo, branch = `main`, main file = `app.py`
4. Click **Deploy**

Your app will be live in about a minute.

### 3. Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the project tests

**Question:** Does buying during market fear (and selling during greed) actually make more money?

**Method:** Linear regression — same as in class.

- **Page 3 (Trade Evidence):** `Closed_PnL ~ FG_value × Side` — tests if the effect
  of sentiment is different for buyers vs sellers.
- **Page 4 (Bitcoin Evidence):** `BTC_Return_next_day ~ FG_value` — tests if today's
  sentiment predicts tomorrow's BTC price.

**Result:** Strong evidence at the trade level, weaker at the daily price level.
Fear really does favour buyers — but not strongly enough to show up in 2 years
of daily BTC prices alone.
