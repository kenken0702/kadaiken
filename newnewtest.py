import sys
import numpy as np
import pandas as pd
import yfinance as yf

# ================================
# 依存ライブラリの事前チェック
# ================================
MISSING = []
for lib in ["yfinance", "pandas", "numpy"]:
    try:
        __import__(lib)
    except ModuleNotFoundError:
        MISSING.append(lib)

if MISSING:
    print("必要ライブラリが不足しています:", ", ".join(MISSING))
    print("次を実行してください: pip install yfinance pandas numpy")
    sys.exit(1)


# ================================
# 設定 (CONFIG)
# ================================
CONFIG = {
    "start_date": "2014-01-01",
    "end_date": "2023-12-31",
    "total_budget_jp": 6000000,
    "total_budget_us": 60000,
    "tickers": ["8088.T"],
    "use_dow_shield": True,
    
    # 動的制御のベースパラメータ
    "window_size": 20,
    "trading_days_yr": 252,
    "bb_sd_mult": 2.0,
    "value_drop_ratio": 0.20,
    "risk_mkt_mult": 1.2,
    "max_lots_per_trade": 3,
    
    # 動的閾値 (Dynamic Threshold) のベース
    "T_base": 0.45,
    "gamma": 0.1,
}

def get_lot_size(ticker: str) -> int:
    return 100 if ticker.endswith(".T") else 1

def get_market_config(ticker):
    if ticker.endswith(".T"):
        return {"benchmark": "^N225", "budget": CONFIG["total_budget_jp"], "cur": "¥"}
    return {"benchmark": "^GSPC", "budget": CONFIG["total_budget_us"], "cur": "$"}

def clip(x, low, high):
    return max(low, min(high, x))


# ================================
# 指標計算ユーティリティ
# ================================
def calc_annualized_vol(close_series, window, trading_days):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(trading_days)

def calc_indicators(df, mkt_vol_raw, win, bb_mult, ann_days):
    out = df.copy()
    
    # 市場ボラティリティとその平均（動的閾値で使用）
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["mkt_vol_avg"] = out["mkt_vol"].expanding().mean()
    
    out["vol"] = calc_annualized_vol(out["Close"], window=win, trading_days=ann_days)
    out["ema"] = out["Close"].ewm(span=win, adjust=False).mean()
    out["bb_sd"] = out["Close"].rolling(window=win).std()
    out["bb_low"] = out["ema"] - (out["bb_sd"] * bb_mult)

    # ダウ・シールド（波の判定）
    if "High" in out.columns and "Low" in out.columns:
        out["recent_high"] = out["High"].rolling(window=win).max()
        out["prev_high"] = out["recent_high"].shift(win)
        out["recent_low"] = out["Low"].rolling(window=win).min()
        out["prev_low"] = out["recent_low"].shift(win)
        out["dow_downtrend"] = (out["recent_high"] < out["prev_high"]) & (out["recent_low"] < out["prev_low"])
    else:
        out["dow_downtrend"] = False
        
    return out

def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))

def build_monthly_contribution_schedule(df, total_budget):
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    month_groups = get_month_groups(df)
    n_months = len(month_groups)
    monthly_budget = total_budget / n_months if n_months > 0 else 0
    
    for _, m_df in month_groups:
        if len(m_df) > 0:
            schedule.loc[m_df.index[0]] = monthly_budget
    return schedule


# ================================
# 【コア】動的適応型 ロジック
# ================================
def execute_dynamic_adaptive_dva(df, contrib_schedule, lot_size, max_lots):
    shares, spent, cash, contribution, trade_count = 0, 0.0, 0.0, 0.0, 0
    
    # 月末（最終営業日）の判定セットを作成
    month_groups = get_month_groups(df)
    last_days = set(m.index[-1] for _, m in month_groups if len(m) > 0)
    
    v_drop = CONFIG["value_drop_ratio"]
    r_mult = CONFIG["risk_mkt_mult"]

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        if pd.isna(price) or price <= 0: continue
            
        ema = df.at[dt, "ema"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]
        mkt_vol_avg = df.at[dt, "mkt_vol_avg"]
        is_dow_downtrend = bool(df.at[dt, "dow_downtrend"])

        # 資金拠出
        add_cash = float(contrib_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        # --------------------------------------------------
        # 1. 動的ウェイト (Dynamic Weights)
        # --------------------------------------------------
        # 下降トレンド時: [焦り: 10%, 割安: 40%, BB: 25%, トレンド: 5%, リスク: 20%]
        # 上昇・通常時: [焦り: 50%, 割安: 20%, BB: 5%, トレンド: 15%, リスク: 10%]
        if is_dow_downtrend:
            w = {"shortage": 0.10, "value": 0.40, "bb": 0.25, "trend": 0.05, "risk": 0.20}
        else:
            w = {"shortage": 0.50, "value": 0.20, "bb": 0.05, "trend": 0.15, "risk": 0.10}

        # --------------------------------------------------
        # 2. スコア計算
        # --------------------------------------------------
        ideal_shares = contribution / ema if (pd.notna(ema) and ema > 0) else 0.0
        shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0) if ideal_shares > 0 else 0.0
        value = clip((ema - price) / (v_drop * ema), 0.0, 1.0) if (pd.notna(ema) and ema > 0) else 0.0
        bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0
        risk = clip((vol / (r_mult * mkt_vol)) - 1.0, 0.0, 1.0) if (pd.notna(vol) and mkt_vol > 0) else 0.0
        trend = 1.0 if (pd.notna(ema) and price < ema) else 0.0

        score = (w["shortage"]*shortage + w["value"]*value + w["bb"]*bb + w["trend"]*trend - w["risk"]*risk)
        
        if CONFIG["use_dow_shield"] and is_dow_downtrend: 
            score = 0.0  # ダウ・シールド発動時はスコアゼロ（見送り）
        score = clip(score, 0.0, 1.0)

        # --------------------------------------------------
        # 3. 動的閾値 (Dynamic Threshold: T_t)
        # --------------------------------------------------
        if pd.notna(mkt_vol) and pd.notna(mkt_vol_avg) and mkt_vol_avg > 0:
            T_t = CONFIG["T_base"] + CONFIG["gamma"] * ((mkt_vol / mkt_vol_avg) - 1.0)
        else:
            T_t = CONFIG["T_base"]
        T_t = clip(T_t, 0.1, 0.9)

        # --------------------------------------------------
        # 4. 動的合成係数 (Dynamic Lambda: lambda_t)
        # --------------------------------------------------
        cash_ratio = (cash / contribution) if contribution > 0 else 1.0
        lambda_t = 1.0 / (1.0 + np.exp(-10.0 * (cash_ratio - 0.5))) # シグモイド関数

        # --------------------------------------------------
        # 5. 購入判定と執行
        # --------------------------------------------------
        lot_cost = price * lot_size
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 else 0
        buy_lots = 0

        if affordable_lots >= 1 and score >= T_t:
            q_score = clip((score - T_t) / (1.0 - T_t), 0.0, 1.0)
            q_cash = clip((affordable_lots - 1) / (max_lots - 1), 0.0, 1.0) if max_lots > 1 else 0.0
            
            # lambda_t を用いた合成
            q_exec = lambda_t * q_score + (1.0 - lambda_t) * q_cash
            wish_lots = 1 + int(np.floor(q_exec * (max_lots - 1)))
            buy_lots = min(affordable_lots, wish_lots)

        if buy_lots > 0:
            buy_sh = buy_lots * lot_size
            cost = buy_sh * price
            shares += buy_sh
            spent += cost
            cash -= cost
            trade_count += 1

        # --------------------------------------------------
        # 6. 月末スイープ機能 (月末残金DCA強制買付)
        # --------------------------------------------------
        if dt in last_days and cash >= lot_cost and price > 0:
            sweep_lots = int(cash // lot_cost)
            if sweep_lots > 0:
                buy_sh = sweep_lots * lot_size
                cost = buy_sh * price
                shares += buy_sh
                spent += cost
                cash -= cost
                trade_count += 1

    return {"shares": shares, "spent": spent, "cash": cash, "contrib": contribution, "trades": trade_count}


# ================================
# ベースライン (毎月月初DCA)
# ================================
def execute_monthly_first(df, contrib_schedule, lot_size):
    shares, spent, cash, contribution, trade_count = 0, 0.0, 0.0, 0.0, 0
    for dt in df.index:
        price = float(df.at[dt, "Close"])
        add_cash = float(contrib_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        if add_cash > 0 and not pd.isna(price) and price > 0:
            lot_cost = price * lot_size
            buy_lots = int(cash // lot_cost)
            if buy_lots > 0:
                buy_sh = buy_lots * lot_size
                cost = buy_sh * price
                shares += buy_sh
                spent += cost
                cash -= cost
                trade_count += 1
    return {"shares": shares, "spent": spent, "cash": cash, "contrib": contribution, "trades": trade_count}


# ================================
# 結果表示ユーティリティ
# ================================
def print_results(ticker, cur, res_dynamic, res_dca):
    def format_row(name, r):
        avg = r['spent']/r['shares'] if r['shares'] > 0 else 0
        return (f"{name:<20} | {r['shares']:>10,} 株 | {cur}{avg:>12,.2f} | "
                f"{cur}{r['spent']:>14,.0f} | {cur}{r['cash']:>10,.0f} | {r['trades']:>5} 回")

    print(f"\n【 {ticker} 】 (単元: {get_lot_size(ticker)}株 / 拠出累計: {cur}{res_dca['contrib']:,.0f})")
    print("-" * 90)
    print(f"{'手法名':<20} | {'最終取得株数':>12} | {'平均取得単価':>13} | {'約定総額':>15} | {'繰越現金':>11} | {'売買回数'}")
    print("-" * 90)
    print(format_row("★ 動的適応型 DVA", res_dynamic))
    print(format_row("  ベースライン (DCA)", res_dca))
    print("-" * 90)


# ================================
# メイン実行関数
# ================================
def main():
    print(f"=== 動的適応型DVA (Dynamic Adaptive DVA) シミュレーション ===")
    print(f"期間: {CONFIG['start_date']} 〜 {CONFIG['end_date']}\n")
    
    benchmark_cache = {}
    
    for ticker in CONFIG["tickers"]:
        cfg = get_market_config(ticker)
        bmk = cfg["benchmark"]
        
        if bmk not in benchmark_cache:
            bmk_df = yf.download(bmk, start=CONFIG["start_date"], end=CONFIG["end_date"], auto_adjust=True, progress=False)
            if isinstance(bmk_df.columns, pd.MultiIndex): bmk_df.columns = bmk_df.columns.get_level_values(0)
            benchmark_cache[bmk] = calc_annualized_vol(bmk_df["Close"], CONFIG["window_size"], CONFIG["trading_days_yr"])
            
        df = yf.download(ticker, start=CONFIG["start_date"], end=CONFIG["end_date"], auto_adjust=True, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        df = calc_indicators(
            df, 
            mkt_vol_raw=benchmark_cache[bmk], 
            win=CONFIG["window_size"], 
            bb_mult=CONFIG["bb_sd_mult"], 
            ann_days=CONFIG["trading_days_yr"]
        )
        
        contrib_sched = build_monthly_contribution_schedule(df, cfg["budget"])
        lot_size = get_lot_size(ticker)
        max_lots = CONFIG["max_lots_per_trade"]
        
        # シミュレーション実行
        res_dynamic = execute_dynamic_adaptive_dva(df, contrib_sched, lot_size, max_lots)
        res_dca = execute_monthly_first(df, contrib_sched, lot_size)
        
        print_results(ticker, cfg["cur"], res_dynamic, res_dca)

if __name__ == "__main__":
    main()
