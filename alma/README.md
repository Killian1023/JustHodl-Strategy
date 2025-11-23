# ALMA Strategy (JustHodl)

## 1. High‑Level Overview

This folder contains the **ALMA (Arnaud Legoux Moving Average) swing strategy** used in the Web3 Trading Competition.

The core idea:
- **Timeframe:** trade on a 15‑minute base chart.
- **Signal:** use an **ALMA moving average on a higher “ALT” timeframe** as a trend filter.
- **Direction:** **long‑only**, trend‑following.
- **Risk management:**
  - Entry when price breaks **above** the ALMA line.
  - Exit via a combination of **take‑profit (TP)** and **stop‑loss (SL)**.
  - SL is based on a **recent swing low buffered by ATR**, TP is set using a **fixed risk‑reward multiple**.

All important numerical knobs (ALMA settings, RR, ATR settings, lookback windows, etc.) are **automatically tuned per symbol using a Genetic Algorithm (GA)** on historical data. Only the final per‑symbol parameter sets are used live.

---

## 2. Strategy Logic

### 2.1 Timeframes

- **Base timeframe:** 15 minutes.
- **ALT timeframe:** `ALT = base_tf × alt_multiplier`, where `alt_multiplier` is a per‑symbol parameter.
- Price is aggregated to both timeframes and ALMA is computed on the ALT timeframe.
- The ALT‑timeframe ALMA values are then **forward‑filled** back onto the base timeframe to avoid repainting.

### 2.2 ALMA Trend Filter

For each symbol we maintain a set of ALMA parameters:
- **`alt_multiplier`** – controls how slow the trend filter is (e.g. 3h, 4h, … on top of 15m base).
- **`basis_len`** – ALMA window length.
- **`offset_alma`** – where the ALMA puts more weight inside the window (0–1).
- **`offset_sigma`** – controls how tight or flat the ALMA weighting curve is.

Conceptually, ALMA smooths price while remaining responsive. By running it on a higher timeframe and projecting it down to the trading timeframe, we obtain a **stable, low‑noise trend line**.

### 2.3 Entry & Exit Rules

- **Entry (long only):**
  - Wait until the ALT‑timeframe ALMA updates (i.e. the higher‑timeframe bar closes).
  - On the base timeframe, **enter long** when price **crosses above** the ALMA signal.
  - Orders are executed on the **next bar’s open** (both in backtest and live) to avoid lookahead.

- **Stop‑Loss (SL):**
  - Start from the **lowest low of the last N bars** (`last_n_len`).
  - Subtract an **ATR‑based buffer**: `SL = swing_low − ATR × atr_mult`.
  - If this would end up above entry or invalid, we fall back to a small, conservative buffer below entry.

- **Take‑Profit (TP):**
  - Risk per trade is `risk = entry_price − stop_loss`.
  - TP is placed at `TP = entry_price + rr_multiplier × risk`.

- **Exit:**
  - If price hits **TP** or **SL** intrabar, the position is closed on the next bar’s open in backtest; in live trading, TP/SL are managed by a combination of limit order (for TP) and price monitoring / market order (for SL).
  - If the trend breaks (e.g. price crosses below the signal) without hitting TP/SL, the strategy closes the position.

This gives us a **simple but robust trend‑following system**: enter on confirmed trend continuation, and let volatility‑aware stops plus a fixed risk‑reward structure manage exits.

---

## 3. Risk & Portfolio Management

Risk settings are configured per environment but the key ideas are:

- **Initial capital:** defined in config (e.g. USD 50,000 for live portfolio).
- **Max concurrent positions:** cap on number of open ALMA trades at once.
- **Per‑position size:** fixed percentage of total equity per trade (e.g. ~24% per position).
- **Exchange fees:** modeled via basis‑points per side (e.g. 10 bps) in both backtest and live.
- **Per‑symbol tick sizes:** each market has specific **quantity and price steps**, enforced in live trading so that all orders are valid for the exchange.

There is also a **minimum position notional** to avoid opening dust trades, and the SL/TP model ensures every trade has a well‑defined risk.

---

## 4. Genetic Algorithm Parameter Search (Conceptual)

We use a **Genetic Algorithm (GA)** to determine good parameter sets for each symbol. Only the **resulting parameters and performance summaries** are used by the trading system; the optimization engine remains internal.

### 4.1 GA Objective

For each symbol, the GA searches over:
- Timeframe structure (`alt_multiplier`).
- ALMA shape (`basis_len`, `offset_sigma`, `offset_alma`).
- Risk model (`last_n_len`, `rr_multiplier`).
- ATR swing stop parameters (`atr_len`, `atr_mult`).

Candidate parameter sets are evaluated on **historical 1‑minute data** that is resampled to the base & ALT timeframes. The fitness function combines:
- **Total return** over the training window.
- **Profit factor** (gross profit / gross loss).
- **Risk‑adjusted metrics**, such as Calmar or drawdown‑adjusted growth.
- Penalties for:
  - Too few trades (to avoid overfitting to a handful of lucky trades).
  - **High concentration** of P&L in just a few trades.
  - High variation in performance across time segments.

### 4.2 Robustness & Validation

To reduce overfitting, the GA uses several robustness ideas conceptually:

- **Temporal splits:** data is split into segments / folds; candidate parameters must perform **consistently** across these folds, not only on one period.
- **Mean–variance fitness:** the fitness trades off **average performance** vs **stability** (penalizing high standard deviation of performance across folds).
- **Perturbation tests:** the top candidate in each generation is randomly perturbed; if small parameter changes destroy performance, that solution is penalized.
- **Holdout test:** after optimization, the final candidate is tested on a **reserved out‑of‑sample period** (not seen during GA training) to produce a more realistic performance estimate.

The final output per symbol is a compact JSON/record like:
- **`best_params`**: the chosen ALMA + risk model settings.
- **`train_summary`**: metrics over the training window.
- **`holdout_summary`**: metrics on the final unseen segment.

Live trading then **only consumes `best_params`** for each symbol.

---

## 5. Live Trading Setup

The live trader is designed to mirror the backtest as closely as possible while integrating with a crypto exchange API.

Core ideas:

- **Data flow:**
  - Polls high‑frequency price updates (e.g. 1‑second ticks), aggregates them to the 15‑minute base timeframe, and builds the higher ALT timeframe from that.
  - Reuses the same ALMA and ATR logic as the backtester for signal generation.

- **Execution:**
  - **Market orders** for entries and emergency exits.
  - **Limit orders** for take‑profit levels when possible.
  - Strict rounding to per‑symbol quantity/price steps.

- **Account integration:**
  - Periodically syncs balances from the exchange.
  - Derives **total equity** and **available cash**.
  - Uses the configured portfolio rules to size trades and cap concurrent positions.

- **Logging & monitoring:**
  - Local logging of signals, orders, fills, and P&L.
  - Optional external logging (e.g. database / dashboard) for live monitoring and competition reporting.

This setup ensures that what we see in backtests is as close as possible to what we execute live, subject to normal market frictions (fees, slippage, liquidity).

---

## 6. Intended Use in the Competition

- The ALMA strategy is **fully systematic**: once parameters and symbols are selected, all entries and exits are rule‑driven.
- Per‑symbol parameters are **discovered offline by a GA**, and only the resulting configurations are used in production.
- The live bot:
  - Focuses on a curated list of symbols where the GA found **robust, out‑of‑sample‑validated** performance.
  - Uses conservative risk sizing, volatility‑aware stops, and strict execution rules.

This combination of **trend‑following structure**, **genetic parameter search**, and **risk‑aware live execution** is what drives the performance of our ALMA strategy in the Web3 Trading Competition.
