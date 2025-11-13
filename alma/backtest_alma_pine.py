import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp


@dataclass
class Params:
    csv_path: str
    base_tf_min: int = 15                 # Pine input.res default "15"
    # Fixed strategy settings (always-on):
    use_alt: bool = True                  # always use alternate signals
    alt_multiplier: int = 8               # intRes (subject to GA)
    basis_type: str = "ALMA"              # always ALMA
    basis_len: int = 2                    # basisLen (subject to GA)
    offset_sigma: int = 5                 # ALMA sigma (subject to GA)
    offset_alma: float = 0.85             # ALMA offset m (subject to GA)
    delay_offset: int = 0                 # delay open/close series offset

    # Execution & risk (fixed / always-on long-only):
    initial_capital: float = 10000.0
    fee_bps: float = 10.0                 # fixed: 10 bps per side
    slippage_bps: float = 0.0
    qty_pct: float = 100.0                # fixed: 100% of equity
    # New TP/SL model: SL = lowest low of last_n_len bars; TP = entry + rr*(entry - SL)
    last_n_len: int = 10                  # subject to GA
    rr_multiplier: float = 1.5            # subject to GA
    # Stop model (new): "swing" (default) or "atr_swing" (volatility-buffered swing)
    stop_mode: str = "swing"
    atr_len: int = 14
    atr_mult: float = 1.0


@dataclass
class Trade:
    side: str  # 'long' | 'short'
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


# ------- Data utils -------

def read_1m(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'open_time_iso' in df.columns:
        dt = pd.to_datetime(df['open_time_iso'], utc=True)
    elif 'open_time' in df.columns:
        dt = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    else:
        time_col = next((c for c in df.columns if 'time' in c), None)
        if time_col is None:
            raise ValueError('No timestamp column found (expected open_time_iso or open_time).')
        dt = pd.to_datetime(df[time_col], utc=True)
    df = df.assign(datetime=dt).set_index('datetime')
    needed = ['open', 'high', 'low', 'close', 'volume']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'Missing column: {c}')
    df = df[needed].astype(float)
    return df


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    ohlc = df[['open', 'high', 'low', 'close']].resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })
    vol = df[['volume']].resample(rule).sum()
    out = pd.concat([ohlc, vol], axis=1).dropna()
    out = out[~(out[['open','high','low','close']].isna().any(axis=1))]
    return out


# ------- Indicators matching Pine options -------

def wma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ma(price: pd.Series, n: int) -> pd.Series:
    if n <= 1:
        return price.copy()
    wmaf = wma(price, max(1, round(n * 0.5)))
    wmas = wma(price, n)
    return wma(2.0 * wmaf - wmas, max(1, round(math.sqrt(n))))


def tema(src: pd.Series, length: int) -> pd.Series:
    ema1 = src.ewm(span=length, adjust=False, min_periods=length).mean()
    ema2 = ema1.ewm(span=length, adjust=False, min_periods=length).mean()
    ema3 = ema2.ewm(span=length, adjust=False, min_periods=length).mean()
    return 3 * (ema1 - ema2) + ema3


def alma(src: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    # Pine ta.alma uses m (offset [0..1]) and sigma (usually 6) over a window length
    if length <= 1:
        return src.copy()
    m = int(math.floor(offset * (length - 1)))
    s = length / sigma if sigma != 0 else length
    def alma_row(x: np.ndarray) -> float:
        w = np.exp(-((np.arange(length) - m) ** 2) / (2 * (s ** 2)))
        return float(np.dot(w, x) / np.sum(w))
    return src.rolling(length).apply(lambda x: alma_row(x.values), raw=False)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    if length <= 1:
        return (df['high'] - df['low']).abs()
    tr = true_range(df['high'], df['low'], df['close'])
    # Use Wilder's smoothing approximation via EMA with alpha=1/length
    return tr.ewm(alpha=1.0/float(max(1, length)), adjust=False, min_periods=length).mean()


def linreg(src: pd.Series, length: int, offset: int) -> pd.Series:
    # Least squares moving average similar to Pine's ta.linreg
    if length <= 1:
        return src.copy()
    x = np.arange(length)
    def lr(y):
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]  # y ~ a*x + b
        return a * (length - 1 - offset) + b
    return src.rolling(length).apply(lambda y: lr(y.values), raw=True)


def variant(basis_type: str, src: pd.Series, length: int, offset_sigma: int, offset_alma: float) -> pd.Series:
    basis_type = basis_type.upper()
    if basis_type == 'TEMA':
        return tema(src, max(1, length))
    elif basis_type == 'HULLMA':
        return hull_ma(src, max(1, length))
    elif basis_type == 'ALMA':
        # offset_sigma maps to sigma, offset_alma maps to offset m in [0..1]
        sigma = max(1e-6, float(offset_sigma))
        return alma(src, max(1, length), float(offset_alma), sigma)
    else:
        # Fallback to EMA if unknown
        return src.ewm(span=max(1, length), adjust=False).mean()


def crossover(a_now: float, b_now: float, a_prev: float, b_prev: float) -> bool:
    return (a_prev is not None and b_prev is not None and a_prev <= b_prev and a_now > b_now)


def crossunder(a_now: float, b_now: float, a_prev: float, b_prev: float) -> bool:
    return (a_prev is not None and b_prev is not None and a_prev >= b_prev and a_now < b_now)


# ------- Strategy logic -------

def prepare_series(base: pd.DataFrame, alt: pd.DataFrame, p: Params) -> Tuple[pd.Series, pd.Series]:
    # delay offset on base series
    close_base = base['close'].shift(p.delay_offset)
    open_base = base['open'].shift(p.delay_offset)
    if p.use_alt:
        # Compute variant on ALT timeframe, then bring back to base by forward-fill (no repaint)
        close_alt = variant(p.basis_type, alt['close'], p.basis_len, p.offset_sigma, p.offset_alma)
        open_alt = variant(p.basis_type, alt['open'], p.basis_len, p.offset_sigma, p.offset_alma)
        # Align to base: reindex and ffill so higher-tf value holds until next close
        close_alt_on_base = close_alt.reindex(base.index, method='pad')
        open_alt_on_base = open_alt.reindex(base.index, method='pad')
        return close_alt_on_base, open_alt_on_base
    else:
        # Use base timeframe directly
        return (
            variant(p.basis_type, close_base, p.basis_len, p.offset_sigma, p.offset_alma),
            variant(p.basis_type, open_base, p.basis_len, p.offset_sigma, p.offset_alma),
        )


def backtest_with_data(base: pd.DataFrame, alt: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, List[Trade], float]:
    # Compute Pine-like alt series used for signals on the base chart
    closeS, openS = prepare_series(base, alt, p)

    result = base.copy()
    result['closeSeriesAlt'] = closeS
    result['openSeriesAlt'] = openS

    fee = p.fee_bps * 1e-4
    slip = p.slippage_bps * 1e-4

    trades: List[Trade] = []
    equity = p.initial_capital
    position: Optional[str] = None  # 'long'|None (long-only)
    qty = 0.0
    entry_price = math.nan
    current_sl = math.nan
    current_tp = math.nan

    idx = list(result.index)

    # Pre-compute ATR on base timeframe for volatility-buffered SL if requested
    atr_series = atr(result, max(2, int(p.atr_len))) if getattr(p, 'stop_mode', 'swing') == 'atr_swing' else None

    # Detect bars where ALT value updated (i.e., higher TF closed)
    alt_changed = (result['closeSeriesAlt'] != result['closeSeriesAlt'].shift(1)) | (result['openSeriesAlt'] != result['openSeriesAlt'].shift(1))

    pending_entry: Optional[pd.Timestamp] = None  # execute_at_ts
    pending_exit: Optional[pd.Timestamp] = None

    for i, ts in enumerate(idx):
        open_px = float(result['open'].iloc[i])
        high_px = float(result['high'].iloc[i])
        low_px = float(result['low'].iloc[i])

        # 1) Execute pending exit at this bar open
        if pending_exit is not None and ts == pending_exit and position is not None:
            filled = open_px * (1 - slip)
            fee_exit = fee * filled * qty
            pnl = (filled - entry_price) * qty - fee_exit
            equity += pnl
            trades[-1].exit_time = ts
            trades[-1].exit_price = filled
            trades[-1].pnl = pnl
            position = None
            qty = 0.0
            entry_price = math.nan
            current_sl = math.nan
            current_tp = math.nan
            pending_exit = None

        # 2) Execute pending entry at this bar open
        if pending_entry is not None and ts == pending_entry and position is None:
            filled = open_px * (1 + slip)
            deploy = max(0.0, equity * (p.qty_pct / 100.0))
            q = 0.0 if filled <= 0 else deploy / filled
            if q > 0:
                fee_entry = fee * filled * q
                equity -= fee_entry
                qty = q
                entry_price = filled
                position = 'long'
                trades.append(Trade('long', ts, filled, qty))
                # Set SL/TP at entry using selected stop model over PREVIOUS bars (exclude current bar to avoid lookahead)
                lo_start = max(0, i - p.last_n_len)
                window_low = float(result['low'].iloc[lo_start:i].min())
                if getattr(p, 'stop_mode', 'swing') == 'atr_swing' and atr_series is not None:
                    # Use ATR from previous bar to avoid lookahead
                    atr_val = float(atr_series.iloc[i - 1]) if i - 1 >= 0 and math.isfinite(float(atr_series.iloc[i - 1])) else 0.0
                    current_sl = window_low - float(p.atr_mult) * max(0.0, atr_val)
                else:
                    current_sl = window_low
                # If SL >= entry or invalid, set a minimal buffer to avoid invalid RR
                if not math.isfinite(current_sl) or current_sl >= entry_price:
                    # fallback: nudge below entry by small fraction of price or ATR
                    fallback = (float(p.atr_mult) * float(atr_series.iloc[i - 1]) if atr_series is not None and i - 1 >= 0 and math.isfinite(float(atr_series.iloc[i - 1])) else entry_price * 0.005)
                    current_sl = entry_price - max(1e-9, fallback)
                rr = max(0.01, float(p.rr_multiplier))
                current_tp = entry_price + rr * (entry_price - current_sl)
            pending_entry = None

        # 3) Update exits: TP/SL intrabar check; exit next bar open
        if position is not None:
            hit_tk = math.isfinite(current_tp) and high_px >= current_tp
            hit_sl = math.isfinite(current_sl) and low_px <= current_sl
            if hit_tk or hit_sl:
                pending_exit = idx[i + 1] if i + 1 < len(idx) else ts

        # 4) Signal generation only when the higher timeframe updates (non-repainting intent)
        if not bool(alt_changed.iloc[i]):
            continue

        a_now = result['closeSeriesAlt'].iloc[i]
        b_now = result['openSeriesAlt'].iloc[i]
        a_prev = result['closeSeriesAlt'].iloc[i - 1] if i > 0 else None
        b_prev = result['openSeriesAlt'].iloc[i - 1] if i > 0 else None

        buy_sig = crossover(a_now, b_now, a_prev, b_prev)

        # Flip logic like Pine (pyramiding=0): if in long and new buy signal appears, do nothing.
        # If flat and buy signal: schedule entry next bar open.
        next_ts = idx[i + 1] if i + 1 < len(idx) else ts
        if buy_sig and position is None:
            pending_entry = next_ts

    # Close any open position at last close
    if position is not None:
        last_ts = idx[-1]
        last_close = float(result['close'].iloc[-1])
        filled = last_close
        fee_exit = fee * filled * qty
        pnl = (filled - entry_price) * qty - fee_exit
        equity += pnl
        trades[-1].exit_time = last_ts
        trades[-1].exit_price = filled
        trades[-1].pnl = pnl

    return result, trades, equity


def backtest(p: Params) -> Tuple[pd.DataFrame, List[Trade], float]:
    raw_1m = read_1m(p.csv_path)
    base = resample_ohlcv(raw_1m, p.base_tf_min)
    alt_minutes = p.base_tf_min * (p.alt_multiplier if p.use_alt else 1)
    alt = resample_ohlcv(raw_1m, alt_minutes)
    return backtest_with_data(base, alt, p)


def summarize(trades: List[Trade], initial_capital: float, final_equity: float, 
              period_start: Optional[pd.Timestamp] = None, period_end: Optional[pd.Timestamp] = None) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if (t.pnl or 0) > 0)
    total_pnl = sum(t.pnl or 0 for t in trades)
    winrate = (wins / n * 100.0) if n > 0 else 0.0
    ret_pct = (final_equity - initial_capital) / initial_capital * 100.0

    # Profit Factor
    gross_profit = sum((t.pnl or 0) for t in trades if (t.pnl or 0) > 0)
    gross_loss = -sum((t.pnl or 0) for t in trades if (t.pnl or 0) < 0)
    if gross_loss <= 0 and gross_profit > 0:
        # No losing trades: use a reasonable cap instead of infinity
        profit_factor = 100.0
    elif gross_loss <= 0:
        profit_factor = 0.0
    else:
        profit_factor = gross_profit / gross_loss

    # Calmar Ratio (CAGR / MaxDrawdown)
    # Build equity curve at trade exits (stepwise). If no trades, dd=0 and calmar=0.
    equity_points: List[Tuple[pd.Timestamp, float]] = []
    eq = initial_capital
    for t in trades:
        if t.exit_time is not None and t.pnl is not None:
            eq += t.pnl
            equity_points.append((t.exit_time, eq))
    # Determine start/end for CAGR
    if period_start is None and trades:
        period_start = trades[0].entry_time
    if period_end is None and trades:
        period_end = trades[-1].exit_time or trades[-1].entry_time
    # Max drawdown from equity_points (fallback to ret if empty)
    max_dd = 0.0
    if equity_points:
        peak = equity_points[0][1]
        for _, v in equity_points:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    years = 0.0
    if period_start is not None and period_end is not None:
        dt_years = (period_end - period_start).total_seconds() / (365.25 * 24 * 3600)
        years = max(dt_years, 1e-6)
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) if years > 0 and initial_capital > 0 and final_equity > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 1e-9 else float('inf') if cagr > 0 else 0.0

    return {
        'trades': n,
        'wins': wins,
        'winrate_pct': round(winrate, 2),
        'total_pnl_quote': round(total_pnl, 2),
        'final_equity': round(final_equity, 2),
        'return_pct': round(ret_pct, 2),
        'profit_factor': float(profit_factor),
        'calmar': float(calmar if calmar != float('inf') else 1e3),
        'max_drawdown_pct': round(max_dd * 100.0, 2),
        'cagr_pct': round(cagr * 100.0, 2),
    }


# ------------------------
# Simple Genetic Algorithm
# ------------------------

def score_from_summary(summary: dict) -> float:
    ret = float(summary.get('return_pct', 0.0))
    pf = float(summary.get('profit_factor', 0.0))
    calmar = float(summary.get('calmar', 0.0))
    n_trades = int(summary.get('trades', 0))
    # Normalize/clip for stability
    pf_c = min(pf, 10.0)  # cap PF contribution
    calmar_c = min(calmar, 10.0)
    # Weighted fitness: emphasize return and calmar, include PF
    fitness = 0.5 * ret + 0.3 * (pf_c * 5.0) + 0.2 * (calmar_c * 100.0)
    # Small penalty for very low sample size
    if n_trades < 5:
        fitness -= (5 - n_trades) * 2.0
    return fitness

# ---- Parallel evaluation helpers ----
EVAL_CTX = None  # set in workers

def _init_worker(ctx):
    global EVAL_CTX
    EVAL_CTX = ctx

def _month_slices(idx: pd.DatetimeIndex) -> List[pd.DatetimeIndex]:
    months = sorted(set((ts.year, ts.month) for ts in idx))
    out: List[pd.DatetimeIndex] = []
    for y, m in months:
        m_idx = idx[(idx.year == y) & (idx.month == m)]
        if len(m_idx) > 0:
            out.append(m_idx)
    return out

def _monthly_fitness(indiv: "Params", ctx: dict) -> Tuple[float, List[dict]]:
    base_opt: pd.DataFrame = ctx['base_opt']
    raw_opt: pd.DataFrame = ctx['raw_opt']
    opt_month_indices: List[pd.DatetimeIndex] = ctx['opt_month_indices']
    base_tf_min: int = ctx['base_tf_min']
    alpha: float = ctx['alpha']
    trade_cv_weight: float = ctx['trade_cv_weight']
    conc_threshold: float = ctx['conc_threshold']
    conc_weight: float = ctx['conc_weight']

    month_summaries: List[dict] = []
    month_scores: List[float] = []
    trades_per_month: List[int] = []
    for m_idx in opt_month_indices:
        base_m = base_opt[base_opt.index.isin(m_idx)]
        if len(base_m) == 0:
            continue
        alt_minutes = base_tf_min * indiv.alt_multiplier
        raw_m = raw_opt[raw_opt.index.isin(base_m.index)]
        alt_m = resample_ohlcv(raw_m, alt_minutes)
        _, trades_m, eq_m = backtest_with_data(base_m, alt_m, indiv)
        summ_m = summarize(trades_m, indiv.initial_capital, eq_m,
                           period_start=base_m.index[0], period_end=base_m.index[-1])
        trades_per_month.append(int(summ_m.get('trades', 0)))
        score_m = score_from_summary(summ_m)
        month_summaries.append(summ_m)
        month_scores.append(score_m)
    if not month_scores:
        return -1e18, month_summaries
    mean = float(np.mean(month_scores))
    std = float(np.std(month_scores))
    kfold_score = mean - alpha * std
    # Trade count deviation penalty across months (CV)
    if trades_per_month:
        t_mean = np.mean(trades_per_month)
        t_std = np.std(trades_per_month)
        cv = float(t_std / (t_mean + 1e-9))
        kfold_score -= trade_cv_weight * cv
    # Concentration penalty on entire window
    alt_minutes_full = base_tf_min * indiv.alt_multiplier
    alt_full = resample_ohlcv(raw_opt, alt_minutes_full)
    _, trades_full, eq_full = backtest_with_data(base_opt, alt_full, indiv)
    total_pnl = sum((t.pnl or 0) for t in trades_full)
    if total_pnl > 0:
        top10 = sorted([(t.pnl or 0) for t in trades_full], reverse=True)[:10]
        top10_sum = sum(x for x in top10 if x > 0)
        ratio = top10_sum / total_pnl if total_pnl != 0 else 0.0
        if ratio > conc_threshold:
            kfold_score -= conc_weight * (ratio - conc_threshold)
    return kfold_score, month_summaries

def _eval_individual(indiv: "Params") -> Tuple[float, "Params", dict]:
    # Uses global EVAL_CTX in workers
    score_kfold, month_summaries = _monthly_fitness(indiv, EVAL_CTX)
    if month_summaries:
        avg_ret = float(np.mean([m.get('return_pct', 0.0) for m in month_summaries]))
        avg_pf = float(np.mean([m.get('profit_factor', 0.0) for m in month_summaries]))
        avg_calmar = float(np.mean([m.get('calmar', 0.0) for m in month_summaries]))
        avg_trades = int(np.mean([m.get('trades', 0) for m in month_summaries]))
        agg_summ = {
            'return_pct': avg_ret,
            'profit_factor': avg_pf,
            'calmar': avg_calmar,
            'trades': avg_trades,
        }
    else:
        agg_summ = {'return_pct': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'trades': 0}
    return score_kfold, indiv, agg_summ


def run_ga_on_last_3m_opt_last_1m_test(raw_1m: pd.DataFrame, p_base: Params, pop: int = 24, gens: int = 15, seed: int = 42) -> Tuple[Params, dict, dict]:
    rng = np.random.default_rng(seed)

    # Time splits: last month for test, previous three months for optimization
    if len(raw_1m.index) == 0:
        raise ValueError('Empty dataset for GA.')
    end = raw_1m.index[-1]
    test_start = end - pd.DateOffset(months=1)
    opt_start = test_start - pd.DateOffset(months=3)
    raw_opt = raw_1m[(raw_1m.index >= opt_start) & (raw_1m.index < test_start)]
    raw_test = raw_1m[raw_1m.index >= test_start]
    if len(raw_opt) < 1000 or len(raw_test) < 100:
        # Fallback to 120 days and 30 days window sizes if data sparse
        test_start = end - pd.Timedelta(days=30)
        opt_start = test_start - pd.Timedelta(days=90)
        raw_opt = raw_1m[(raw_1m.index >= opt_start) & (raw_1m.index < test_start)]
        raw_test = raw_1m[raw_1m.index >= test_start]

    base_opt = resample_ohlcv(raw_opt, p_base.base_tf_min)
    base_test = resample_ohlcv(raw_test, p_base.base_tf_min)
    # Monthly k-fold split inside the 3-month optimization window
    opt_month_indices = _month_slices(base_opt.index)

    # Parameter bounds
    forced_stop = getattr(p_base, 'force_stop_mode', None)
    def random_params() -> Params:
        p = Params(csv_path=p_base.csv_path, base_tf_min=p_base.base_tf_min)
        p.alt_multiplier = int(rng.integers(2, 17))
        p.basis_len = int(rng.integers(1, 9))
        p.offset_sigma = int(rng.integers(1, 11))
        p.offset_alma = float(rng.uniform(0.05, 0.95))
        p.last_n_len = int(rng.integers(3, 51))
        p.rr_multiplier = float(rng.uniform(0.5, 4.0))
        # New ATR-swing stop params
        p.stop_mode = forced_stop if forced_stop else 'atr_swing'
        p.atr_len = int(rng.integers(7, 29))
        p.atr_mult = float(rng.uniform(0.5, 2.0))
        return p

    def mutate(p: Params) -> Params:
        q = Params(csv_path=p.csv_path, base_tf_min=p.base_tf_min)
        q.alt_multiplier = int(np.clip(p.alt_multiplier + rng.integers(-2, 3), 2, 16))
        q.basis_len = int(np.clip(p.basis_len + rng.integers(-1, 2), 1, 12))
        q.offset_sigma = int(np.clip(p.offset_sigma + rng.integers(-2, 3), 1, 15))
        q.offset_alma = float(np.clip(p.offset_alma + rng.normal(0, 0.05), 0.01, 0.99))
        q.last_n_len = int(np.clip(p.last_n_len + rng.integers(-3, 4), 2, 60))
        q.rr_multiplier = float(np.clip(p.rr_multiplier + rng.normal(0, 0.2), 0.2, 6.0))
        q.stop_mode = forced_stop if forced_stop else getattr(p, 'stop_mode', 'atr_swing')
        q.atr_len = int(np.clip(getattr(p, 'atr_len', 14) + rng.integers(-3, 4), 5, 40))
        q.atr_mult = float(np.clip(getattr(p, 'atr_mult', 1.0) + rng.normal(0, 0.1), 0.2, 3.0))
        return q

    # Build evaluation context for optional multiprocessing
    alpha = getattr(p_base, 'ga_alpha', 0.5) if hasattr(p_base, 'ga_alpha') else 0.5
    trade_cv_weight = getattr(p_base, 'trade_cv_weight', 10.0) if hasattr(p_base, 'trade_cv_weight') else 10.0
    conc_threshold = getattr(p_base, 'conc_threshold', 0.30) if hasattr(p_base, 'conc_threshold') else 0.30
    conc_weight = getattr(p_base, 'conc_weight', 50.0) if hasattr(p_base, 'conc_weight') else 50.0
    ctx = {
        'base_opt': base_opt,
        'raw_opt': raw_opt,
        'opt_month_indices': opt_month_indices,
        'base_tf_min': p_base.base_tf_min,
        'alpha': alpha,
        'trade_cv_weight': trade_cv_weight,
        'conc_threshold': conc_threshold,
        'conc_weight': conc_weight,
    }

    population: List[Params] = [random_params() for _ in range(pop)]
    best_p = None
    best_score = -1e18
    best_summary = {}

    # Controls for robustness
    robust_k = getattr(p_base, 'robust_k', 3) if hasattr(p_base, 'robust_k') else 3
    robust_noise = getattr(p_base, 'robust_noise', 0.1) if hasattr(p_base, 'robust_noise') else 0.1
    mp_workers = getattr(p_base, 'mp_workers', 0) if hasattr(p_base, 'mp_workers') else 0

    pool = None
    if mp_workers and mp_workers > 0:
        # Initialize pool with shared eval context
        try:
            pool = mp.Pool(processes=mp_workers, initializer=_init_worker, initargs=(ctx,))
        except Exception:
            pool = None  # fallback to single-process if Pool fails

    for g in range(1, gens + 1):
        if pool is not None:
            try:
                scores = pool.map(_eval_individual, population)
            except Exception:
                scores = []
                for indiv in population:
                    sc, ms = _monthly_fitness(indiv, ctx)
                    # fallback aggregation
                    if ms:
                        avg_ret = float(np.mean([m.get('return_pct', 0.0) for m in ms]))
                        avg_pf = float(np.mean([m.get('profit_factor', 0.0) for m in ms]))
                        avg_calmar = float(np.mean([m.get('calmar', 0.0) for m in ms]))
                        avg_trades = int(np.mean([m.get('trades', 0) for m in ms]))
                        agg = {'return_pct': avg_ret, 'profit_factor': avg_pf, 'calmar': avg_calmar, 'trades': avg_trades}
                    else:
                        agg = {'return_pct': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'trades': 0}
                    scores.append((sc, indiv, agg))
        else:
            scores = []
            for indiv in population:
                sc, ms = _monthly_fitness(indiv, ctx)
                if ms:
                    avg_ret = float(np.mean([m.get('return_pct', 0.0) for m in ms]))
                    avg_pf = float(np.mean([m.get('profit_factor', 0.0) for m in ms]))
                    avg_calmar = float(np.mean([m.get('calmar', 0.0) for m in ms]))
                    avg_trades = int(np.mean([m.get('trades', 0) for m in ms]))
                    agg = {'return_pct': avg_ret, 'profit_factor': avg_pf, 'calmar': avg_calmar, 'trades': avg_trades}
                else:
                    agg = {'return_pct': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'trades': 0}
                scores.append((sc, indiv, agg))
        scores.sort(key=lambda x: x[0], reverse=True)
        # Per-generation log for top individual
        top_score, top_indiv, top_summ = scores[0]
        print(f"[GA] Gen {g}/{gens}: kfold_score={top_score:.2f} ret~={top_summ.get('return_pct', 0):.2f}% PF~={top_summ.get('profit_factor', 0):.2f} "
              f"Calmar={top_summ.get('calmar', 0):.2f} trades~={top_summ.get('trades', 0)} "
              f"alt_mult={top_indiv.alt_multiplier} basis_len={top_indiv.basis_len} sigma={top_indiv.offset_sigma} "
              f"m={top_indiv.offset_alma:.3f} last_n={top_indiv.last_n_len} rr={top_indiv.rr_multiplier:.2f} "
              f"stop={getattr(top_indiv, 'stop_mode', 'swing')} atr_len={getattr(top_indiv, 'atr_len', 0)} atr_mult={getattr(top_indiv, 'atr_mult', 0.0):.2f}", flush=True)
        # Robustness check: perturb top_indiv and penalize instability
        def perturb_once(ind: Params) -> Params:
            r = Params(csv_path=ind.csv_path, base_tf_min=ind.base_tf_min)
            r.alt_multiplier = int(np.clip(ind.alt_multiplier + rng.integers(-2, 3), 2, 16))
            r.basis_len = int(np.clip(ind.basis_len + rng.integers(-1, 2), 1, 12))
            r.offset_sigma = int(np.clip(ind.offset_sigma + rng.integers(-2, 3), 1, 15))
            r.offset_alma = float(np.clip(ind.offset_alma + rng.normal(0, robust_noise), 0.01, 0.99))
            r.last_n_len = int(np.clip(ind.last_n_len + rng.integers(-3, 4), 2, 60))
            r.rr_multiplier = float(np.clip(ind.rr_multiplier + rng.normal(0, robust_noise), 0.2, 6.0))
            r.stop_mode = forced_stop if forced_stop else getattr(ind, 'stop_mode', 'atr_swing')
            r.atr_len = int(np.clip(getattr(ind, 'atr_len', 14) + rng.integers(-2, 3), 5, 40))
            r.atr_mult = float(np.clip(getattr(ind, 'atr_mult', 1.0) + rng.normal(0, robust_noise / 2), 0.2, 3.0))
            return r

        robust_scores = []
        for _ in range(max(0, robust_k)):
            r_ind = perturb_once(top_indiv)
            r_score, _ = _monthly_fitness(r_ind, ctx)
            robust_scores.append(r_score)
        if robust_scores:
            avg_robust = float(np.mean(robust_scores))
            # Penalize if robust avg significantly below top score
            top_score = top_score - max(0.0, (top_score - avg_robust) * 0.5)
        if top_score > best_score:
            best_score, best_p, best_summary = top_score, top_indiv, top_summ
        # Elitism + mutate top
        elites = [s[1] for s in scores[: max(2, pop // 5)]]
        new_pop = elites.copy()
        while len(new_pop) < pop:
            parent = elites[int(rng.integers(0, len(elites)))]
            child = mutate(parent)
            new_pop.append(child)
        population = new_pop

    # Evaluate best on hold-out last month test
    alt_minutes_best = p_base.base_tf_min * best_p.alt_multiplier
    alt_test = resample_ohlcv(raw_test, alt_minutes_best)
    _, test_trades, test_eq = backtest_with_data(base_test, alt_test, best_p)
    test_summary = summarize(test_trades, best_p.initial_capital, test_eq,
                             period_start=base_test.index[0] if len(base_test.index) else None,
                             period_end=base_test.index[-1] if len(base_test.index) else None)
    if pool is not None:
        pool.close()
        pool.join()

    return best_p, best_summary, test_summary


def _kfold_fitness(indiv: "Params", ctx: dict) -> Tuple[float, List[dict]]:
    base_train: pd.DataFrame = ctx['base_train']
    raw_train: pd.DataFrame = ctx['raw_train']
    fold_base_indices: List[pd.DatetimeIndex] = ctx['fold_base_indices']
    base_tf_min: int = ctx['base_tf_min']
    alpha: float = ctx['alpha']
    trade_cv_weight: float = ctx['trade_cv_weight']
    conc_threshold: float = ctx['conc_threshold']
    conc_weight: float = ctx['conc_weight']

    month_summaries: List[dict] = []
    month_scores: List[float] = []
    trades_per_fold: List[int] = []
    for m_idx in fold_base_indices:
        base_m = base_train[base_train.index.isin(m_idx)]
        if len(base_m) == 0:
            continue
        alt_minutes = base_tf_min * indiv.alt_multiplier
        raw_m = raw_train[raw_train.index.isin(base_m.index)]
        alt_m = resample_ohlcv(raw_m, alt_minutes)
        _, trades_m, eq_m = backtest_with_data(base_m, alt_m, indiv)
        summ_m = summarize(trades_m, indiv.initial_capital, eq_m,
                           period_start=base_m.index[0], period_end=base_m.index[-1])
        trades_per_fold.append(int(summ_m.get('trades', 0)))
        score_m = score_from_summary(summ_m)
        month_summaries.append(summ_m)
        month_scores.append(score_m)
    if not month_scores:
        return -1e18, month_summaries
    mean = float(np.mean(month_scores))
    std = float(np.std(month_scores))
    kfold_score = mean - alpha * std
    if trades_per_fold:
        t_mean = np.mean(trades_per_fold)
        t_std = np.std(trades_per_fold)
        cv = float(t_std / (t_mean + 1e-9))
        kfold_score -= trade_cv_weight * cv
    # Concentration penalty on entire training window
    alt_minutes_full = base_tf_min * indiv.alt_multiplier
    alt_full = resample_ohlcv(raw_train, alt_minutes_full)
    _, trades_full, _ = backtest_with_data(base_train, alt_full, indiv)
    total_pnl = sum((t.pnl or 0) for t in trades_full)
    if total_pnl > 0:
        top10 = sorted([(t.pnl or 0) for t in trades_full], reverse=True)[:10]
        top10_sum = sum(x for x in top10 if x > 0)
        ratio = top10_sum / total_pnl if total_pnl != 0 else 0.0
        if ratio > conc_threshold:
            kfold_score -= conc_weight * (ratio - conc_threshold)
    return kfold_score, month_summaries


def run_ga_on_30_30_30_train_10_test(raw_1m: pd.DataFrame, p_base: Params, pop: int = 24, gens: int = 15, seed: int = 42) -> Tuple[Params, dict, dict]:
    rng = np.random.default_rng(seed)

    if len(raw_1m.index) == 0:
        raise ValueError('Empty dataset for GA.')

    n = len(raw_1m.index)
    cut1 = int(max(1, round(n * 0.30)))
    cut2 = int(max(cut1 + 1, round(n * 0.60)))
    cut3 = int(max(cut2 + 1, round(n * 0.90)))

    raw_fold1 = raw_1m.iloc[:cut1]
    raw_fold2 = raw_1m.iloc[cut1:cut2]
    raw_fold3 = raw_1m.iloc[cut2:cut3]
    raw_test = raw_1m.iloc[cut3:]

    # Training window is the first 90%
    raw_train = raw_1m.iloc[:cut3]
    base_train = resample_ohlcv(raw_train, p_base.base_tf_min)

    # Derive fold base indices by timestamp boundaries of raw folds
    def idx_to_mask(end_ts, start_ts=None):
        if start_ts is None:
            return base_train.index <= end_ts
        return (base_train.index > start_ts) & (base_train.index <= end_ts)

    t1_end = raw_fold1.index[-1] if len(raw_fold1.index) else raw_train.index[0]
    t2_end = raw_fold2.index[-1] if len(raw_fold2.index) else t1_end
    t3_end = raw_fold3.index[-1] if len(raw_fold3.index) else t2_end

    f1_idx = base_train.index[idx_to_mask(t1_end)]
    f2_idx = base_train.index[idx_to_mask(t2_end, t1_end)]
    f3_idx = base_train.index[idx_to_mask(t3_end, t2_end)]
    fold_base_indices = [f1_idx, f2_idx, f3_idx]

    # Build evaluation context
    alpha = getattr(p_base, 'ga_alpha', 0.5) if hasattr(p_base, 'ga_alpha') else 0.5
    trade_cv_weight = getattr(p_base, 'trade_cv_weight', 10.0) if hasattr(p_base, 'trade_cv_weight') else 10.0
    conc_threshold = getattr(p_base, 'conc_threshold', 0.30) if hasattr(p_base, 'conc_threshold') else 0.30
    conc_weight = getattr(p_base, 'conc_weight', 50.0) if hasattr(p_base, 'conc_weight') else 50.0
    ctx = {
        'base_train': base_train,
        'raw_train': raw_train,
        'fold_base_indices': fold_base_indices,
        'base_tf_min': p_base.base_tf_min,
        'alpha': alpha,
        'trade_cv_weight': trade_cv_weight,
        'conc_threshold': conc_threshold,
        'conc_weight': conc_weight,
    }

    # Parameter bounds
    forced_stop = getattr(p_base, 'force_stop_mode', None)
    def random_params() -> Params:
        p = Params(csv_path=p_base.csv_path, base_tf_min=p_base.base_tf_min)
        p.alt_multiplier = int(rng.integers(2, 17))
        p.basis_len = int(rng.integers(1, 9))
        p.offset_sigma = int(rng.integers(1, 11))
        p.offset_alma = float(rng.uniform(0.05, 0.95))
        p.last_n_len = int(rng.integers(3, 51))
        p.rr_multiplier = float(rng.uniform(0.5, 4.0))
        p.stop_mode = forced_stop if forced_stop else 'atr_swing'
        p.atr_len = int(rng.integers(7, 29))
        p.atr_mult = float(rng.uniform(0.5, 2.0))
        return p

    def mutate(p: Params) -> Params:
        q = Params(csv_path=p.csv_path, base_tf_min=p.base_tf_min)
        q.alt_multiplier = int(np.clip(p.alt_multiplier + rng.integers(-2, 3), 2, 16))
        q.basis_len = int(np.clip(p.basis_len + rng.integers(-1, 2), 1, 12))
        q.offset_sigma = int(np.clip(p.offset_sigma + rng.integers(-2, 3), 1, 15))
        q.offset_alma = float(np.clip(p.offset_alma + rng.normal(0, 0.05), 0.01, 0.99))
        q.last_n_len = int(np.clip(p.last_n_len + rng.integers(-3, 4), 2, 60))
        q.rr_multiplier = float(np.clip(p.rr_multiplier + rng.normal(0, 0.2), 0.2, 6.0))
        q.stop_mode = forced_stop if forced_stop else getattr(p, 'stop_mode', 'atr_swing')
        q.atr_len = int(np.clip(getattr(p, 'atr_len', 14) + rng.integers(-3, 4), 5, 40))
        q.atr_mult = float(np.clip(getattr(p, 'atr_mult', 1.0) + rng.normal(0, 0.1), 0.2, 3.0))
        return q

    robust_k = getattr(p_base, 'robust_k', 3) if hasattr(p_base, 'robust_k') else 3
    robust_noise = getattr(p_base, 'robust_noise', 0.1) if hasattr(p_base, 'robust_noise') else 0.1
    mp_workers = getattr(p_base, 'mp_workers', 0) if hasattr(p_base, 'mp_workers') else 0

    population: List[Params] = [random_params() for _ in range(pop)]
    best_p = None
    best_score = -1e18
    best_summary = {}

    pool = None
    if mp_workers and mp_workers > 0:
        try:
            pool = mp.Pool(processes=mp_workers, initializer=_init_worker, initargs=(ctx,))
        except Exception:
            pool = None

    for g in range(1, gens + 1):
        if pool is not None:
            try:
                scores = pool.map(_eval_individual, population)
            except Exception:
                scores = []
                for indiv in population:
                    sc, ms = _kfold_fitness(indiv, ctx)
                    if ms:
                        avg_ret = float(np.mean([m.get('return_pct', 0.0) for m in ms]))
                        avg_pf = float(np.mean([m.get('profit_factor', 0.0) for m in ms]))
                        avg_calmar = float(np.mean([m.get('calmar', 0.0) for m in ms]))
                        avg_trades = int(np.mean([m.get('trades', 0) for m in ms]))
                        agg = {'return_pct': avg_ret, 'profit_factor': avg_pf, 'calmar': avg_calmar, 'trades': avg_trades}
                    else:
                        agg = {'return_pct': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'trades': 0}
                    scores.append((sc, indiv, agg))
        else:
            scores = []
            for indiv in population:
                sc, ms = _kfold_fitness(indiv, ctx)
                if ms:
                    avg_ret = float(np.mean([m.get('return_pct', 0.0) for m in ms]))
                    avg_pf = float(np.mean([m.get('profit_factor', 0.0) for m in ms]))
                    avg_calmar = float(np.mean([m.get('calmar', 0.0) for m in ms]))
                    avg_trades = int(np.mean([m.get('trades', 0) for m in ms]))
                    agg = {'return_pct': avg_ret, 'profit_factor': avg_pf, 'calmar': avg_calmar, 'trades': avg_trades}
                else:
                    agg = {'return_pct': 0.0, 'profit_factor': 0.0, 'calmar': 0.0, 'trades': 0}
                scores.append((sc, indiv, agg))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_indiv, top_summ = scores[0]
        print(f"[GA] Gen {g}/{gens}: kfold_score={top_score:.2f} ret~={top_summ.get('return_pct', 0):.2f}% PF~={top_summ.get('profit_factor', 0):.2f} "
              f"Calmar={top_summ.get('calmar', 0):.2f} trades~={top_summ.get('trades', 0)} "
              f"alt_mult={top_indiv.alt_multiplier} basis_len={top_indiv.basis_len} sigma={top_indiv.offset_sigma} "
              f"m={top_indiv.offset_alma:.3f} last_n={top_indiv.last_n_len} rr={top_indiv.rr_multiplier:.2f} "
              f"stop={getattr(top_indiv, 'stop_mode', 'swing')} atr_len={getattr(top_indiv, 'atr_len', 0)} atr_mult={getattr(top_indiv, 'atr_mult', 0.0):.2f}", flush=True)

        def perturb_once(ind: Params) -> Params:
            r = Params(csv_path=ind.csv_path, base_tf_min=ind.base_tf_min)
            r.alt_multiplier = int(np.clip(ind.alt_multiplier + rng.integers(-2, 3), 2, 16))
            r.basis_len = int(np.clip(ind.basis_len + rng.integers(-1, 2), 1, 12))
            r.offset_sigma = int(np.clip(ind.offset_sigma + rng.integers(-2, 3), 1, 15))
            r.offset_alma = float(np.clip(ind.offset_alma + rng.normal(0, robust_noise), 0.01, 0.99))
            r.last_n_len = int(np.clip(ind.last_n_len + rng.integers(-3, 4), 2, 60))
            r.rr_multiplier = float(np.clip(ind.rr_multiplier + rng.normal(0, robust_noise), 0.2, 6.0))
            r.stop_mode = forced_stop if forced_stop else getattr(ind, 'stop_mode', 'atr_swing')
            r.atr_len = int(np.clip(getattr(ind, 'atr_len', 14) + rng.integers(-2, 3), 5, 40))
            r.atr_mult = float(np.clip(getattr(ind, 'atr_mult', 1.0) + rng.normal(0, robust_noise / 2), 0.2, 3.0))
            return r

        robust_scores = []
        for _ in range(max(0, robust_k)):
            r_ind = perturb_once(top_indiv)
            r_score, _ = _kfold_fitness(r_ind, ctx)
            robust_scores.append(r_score)
        if robust_scores:
            avg_robust = float(np.mean(robust_scores))
            top_score = top_score - max(0.0, (top_score - avg_robust) * 0.5)
        if top_score > best_score:
            best_score, best_p, best_summary = top_score, top_indiv, top_summ
        elites = [s[1] for s in scores[: max(2, pop // 5)]]
        new_pop = elites.copy()
        while len(new_pop) < pop:
            parent = elites[int(rng.integers(0, len(elites)))]
            child = mutate(parent)
            new_pop.append(child)
        population = new_pop

    # Evaluate best on hold-out last 10%
    base_test = resample_ohlcv(raw_test, p_base.base_tf_min)
    alt_minutes_best = p_base.base_tf_min * best_p.alt_multiplier
    alt_test = resample_ohlcv(raw_test, alt_minutes_best)
    _, test_trades, test_eq = backtest_with_data(base_test, alt_test, best_p)
    test_summary = summarize(test_trades, best_p.initial_capital, test_eq,
                             period_start=base_test.index[0] if len(base_test.index) else None,
                             period_end=base_test.index[-1] if len(base_test.index) else None)
    if pool is not None:
        pool.close()
        pool.join()

    return best_p, best_summary, test_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--base_tf_min', type=int, default=15)
    ap.add_argument('--delay_offset', type=int, default=0)
    ap.add_argument('--initial_capital', type=float, default=10000)
    # GA controls
    ap.add_argument('--optimize', type=int, default=0)
    ap.add_argument('--ga_pop', type=int, default=24)
    ap.add_argument('--ga_gens', type=int, default=15)
    ap.add_argument('--mp_workers', type=int, default=0, help='Multiprocessing workers for GA evaluation (0=disabled)')
    # Robustness & validation controls
    ap.add_argument('--ga_alpha', type=float, default=0.5, help='Std penalty weight for monthly k-fold: mean - alpha*std')
    ap.add_argument('--trade_cv_weight', type=float, default=10.0, help='Penalty weight for trade count coefficient of variation across months')
    ap.add_argument('--conc_threshold', type=float, default=0.30, help='PnL concentration threshold for top-10 trades (fraction of total)')
    ap.add_argument('--conc_weight', type=float, default=50.0, help='Penalty weight when top-10 PnL concentration exceeds threshold')
    ap.add_argument('--robust_k', type=int, default=3, help='Perturbation runs for robustness check per generation')
    ap.add_argument('--robust_noise', type=float, default=0.1, help='Std of perturbation noise for robustness')
    args = ap.parse_args()

    p = Params(
        csv_path=args.csv,
        base_tf_min=args.base_tf_min,
        delay_offset=args.delay_offset,
        initial_capital=args.initial_capital,
    )
    # attach GA control knobs to params container for convenience
    setattr(p, 'ga_alpha', args.ga_alpha)
    setattr(p, 'trade_cv_weight', args.trade_cv_weight)
    setattr(p, 'conc_threshold', args.conc_threshold)
    setattr(p, 'conc_weight', args.conc_weight)
    setattr(p, 'robust_k', args.robust_k)
    setattr(p, 'robust_noise', args.robust_noise)
    setattr(p, 'mp_workers', args.mp_workers)

    if args.optimize:
        # Run GA on previous 3 months, test on last month
        raw_1m = read_1m(p.csv_path)
        best_p, best_summary, test_summary = run_ga_on_last_3m_opt_last_1m_test(raw_1m, p, pop=args.ga_pop, gens=args.ga_gens)
        print('Best Parameters (optimized on previous 3 months):')
        print(f"  alt_multiplier: {best_p.alt_multiplier}")
        print(f"  basis_len: {best_p.basis_len}")
        print(f"  offset_sigma: {best_p.offset_sigma}")
        print(f"  offset_alma: {best_p.offset_alma:.3f}")
        print(f"  last_n_len: {best_p.last_n_len}")
        print(f"  rr_multiplier: {best_p.rr_multiplier:.3f}")
        print('Optimization Summary (prev 3 months):')
        for k, v in best_summary.items():
            print(f"  {k}: {v}")
        print('Hold-out Test Summary (last 1 month):')
        for k, v in test_summary.items():
            print(f"  {k}: {v}")
    else:
        result, trades, equity = backtest(p)
        summary = summarize(trades, p.initial_capital, equity)

        print('Summary:')
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print('\nLast 5 trades:')
        for t in trades[-5:]:
            print(f"  {t}")


if __name__ == '__main__':
    main()
