import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# 基本ユーティリティ
# =========================

def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    df = df.sort_index()
    return df


def get_market_config(ticker, total_budget_us=60000, total_budget_jp=6000000):
    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",
            "total_budget": total_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
            "lot_size": 100,
            "max_lots_per_trade": 2,
        }
    else:
        return {
            "market_name": "US",
            "benchmark": "^GSPC",
            "total_budget": total_budget_us,
            "currency_symbol": "$",
            "currency_code": "USD",
            "lot_size": 1,
            "max_lots_per_trade": 5,
        }


def calc_annualized_vol(close_series, window=20):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calc_indicators(df, mkt_vol_raw):
    out = df.copy()
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["vol"] = calc_annualized_vol(out["Close"], window=20)
    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["bb_low"] = out["ema20"] - (out["Close"].rolling(window=20).std() * 2)
    return out


def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))


def calc_monthly_budget_fixed_total(df, total_budget):
    month_groups = get_month_groups(df)
    n_months = len(month_groups)

    if n_months == 0:
        return 0.0, 0

    monthly_budget = total_budget / n_months
    return monthly_budget, n_months


def build_monthly_contribution_schedule(df, total_budget):
    """
    各月の最初の営業日に月予算を入金する
    """
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        if len(m_df) == 0:
            continue
        schedule.loc[m_df.index[0]] = monthly_budget

    return schedule


# =========================
# DVAスコア
# =========================

def clip(x, low, high):
    return max(low, min(high, x))


def calc_dva_score_row(price, ema, bb_low, vol, mkt_vol, cumulative_contribution, shares):
    """
    実用研究向け DVA スコア
    S_t = clip(0.35*Shortage + 0.35*Value + 0.15*BB + 0.15*Trend - 0.20*Risk, 0, 1)
    """

    if pd.isna(price) or price <= 0:
        return 0.0, {
            "shortage": 0.0, "value": 0.0, "bb": 0.0, "trend": 0.0, "risk": 0.0
        }

    # EMA がない場合はスコア計算不能扱い
    if pd.isna(ema) or ema <= 0:
        return 0.0, {
            "shortage": 0.0, "value": 0.0, "bb": 0.0, "trend": 0.0, "risk": 0.0
        }

    # (2) Ideal_t = C_t / EMA_t
    ideal_shares = cumulative_contribution / ema if ema > 0 else 0.0

    # (3) Shortage_t
    if ideal_shares > 0:
        shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0)
    else:
        shortage = 0.0

    # (4) Value_t
    value = clip((ema - price) / (0.20 * ema), 0.0, 1.0)

    # (5) BB_t
    bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0

    # (6) Risk_t
    if pd.isna(vol) or pd.isna(mkt_vol) or mkt_vol <= 0:
        risk = 0.0
    else:
        risk = clip((vol / (1.2 * mkt_vol)) - 1.0, 0.0, 1.0)

    # (7) Trend_t
    trend = 1.0 if price < ema else 0.0

    # (8) Score
    score_raw = (
        0.35 * shortage +
        0.35 * value +
        0.15 * bb +
        0.15 * trend -
        0.20 * risk
    )
    score = clip(score_raw, 0.0, 1.0)

    return score, {
        "shortage": shortage,
        "value": value,
        "bb": bb,
        "trend": trend,
        "risk": risk,
    }


# =========================
# 執行ルール
# =========================

def execute_practical_dva(
    df,
    contribution_schedule,
    lot_size=1,
    score_threshold=0.55,
    max_lots_per_trade=3,
):
    """
    実用研究向け DVA 執行
    - 月初入金
    - 毎営業日スコア計算
    - 現金が1単元分以上あり、スコアが閾値以上なら買う
    - スコアが高いほど1日に多くの単元を買える
    """

    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    trade_logs = []

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        ema = df.at[dt, "ema20"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]

        add_cash = float(contribution_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        score, parts = calc_dva_score_row(
            price=price,
            ema=ema,
            bb_low=bb_low,
            vol=vol,
            mkt_vol=mkt_vol,
            cumulative_contribution=contribution,
            shares=shares,
        )

        lot_cost = price * lot_size if price > 0 else np.inf
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 else 0

        if affordable_lots >= 1 and score >= score_threshold:
            q = clip((score - score_threshold) / (1.0 - score_threshold), 0.0, 1.0)
            wish_lots = 1 + int(np.floor(q * (max_lots_per_trade - 1)))
            buy_lots = min(affordable_lots, wish_lots)
        else:
            buy_lots = 0

        buy_shares = buy_lots * lot_size
        cost = buy_shares * price

        if buy_shares > 0:
            shares += buy_shares
            spent += cost
            cash -= cost
            trade_count += 1

            trade_logs.append({
                "date": dt,
                "price": price,
                "score": score,
                "buy_shares": buy_shares,
                "cost": cost,
                "cash_after": cash,
                "shortage": parts["shortage"],
                "value": parts["value"],
                "bb": parts["bb"],
                "trend": parts["trend"],
                "risk": parts["risk"],
            })

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
        "trade_logs": pd.DataFrame(trade_logs),
    }


def execute_monthly_first(df, contribution_schedule, lot_size=1):
    """
    毎月(月初)ベースライン
    月初入金直後に、買えるだけ買う
    """
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        add_cash = float(contribution_schedule.at[dt])

        cash += add_cash
        contribution += add_cash

        if add_cash > 0 and price > 0:
            lot_cost = price * lot_size
            buy_lots = int(cash // lot_cost)
            buy_shares = buy_lots * lot_size
            cost = buy_shares * price

            if buy_shares > 0:
                shares += buy_shares
                spent += cost
                cash -= cost
                trade_count += 1

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
    }


def execute_monthly_last(df, contribution_schedule, lot_size=1):
    """
    毎月(月末)ベースライン
    月初に入金し、月末営業日にまとめて買う
    """
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    month_groups = get_month_groups(df)
    contribution_schedule = contribution_schedule.reindex(df.index).fillna(0.0)

    for _, m_df in month_groups:
        # 月内で現金を積み上げ
        for dt in m_df.index:
            add_cash = float(contribution_schedule.at[dt])
            cash += add_cash
            contribution += add_cash

        # 月末日に執行
        last_dt = m_df.index[-1]
        price = float(df.at[last_dt, "Close"])

        if price > 0:
            lot_cost = price * lot_size
            buy_lots = int(cash // lot_cost)
            buy_shares = buy_lots * lot_size
            cost = buy_shares * price

            if buy_shares > 0:
                shares += buy_shares
                spent += cost
                cash -= cost
                trade_count += 1

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
    }


# =========================
# 集計表示
# =========================

def format_result_row(name, result, currency_symbol):
    shares = result["shares"]
    spent = result["spent"]
    cash = result["cash"]
    contribution = result["contribution"]
    trade_count = result["trade_count"]

    avg_price = (spent / shares) if shares > 0 else np.nan

    return {
        "手法名": name,
        "取得株数": shares,
        "平均単価": avg_price,
        "約定総額": spent,
        "繰越現金": cash,
        "拠出総額": contribution,
        "売買回数": trade_count,
        "currency": currency_symbol,
    }


def print_results_table(results):
    results = sorted(results, key=lambda x: x["取得株数"], reverse=True)

    print("-" * 132)
    print(f"{'順位':<2} | {'手法名':<12} | {'取得株数':<10} | {'平均単価':<14} | {'約定総額':<14} | {'繰越現金':<14} | {'拠出総額':<14} | {'売買回数'}")
    print("-" * 132)

    for rank, r in enumerate(results, start=1):
        cur = r["currency"]
        shares_str = f"{r['取得株数']:,d} 株"
        avg_str = f"{cur}{r['平均単価']:,.2f}" if pd.notna(r["平均単価"]) else "-"
        spent_str = f"{cur}{r['約定総額']:,.0f}"
        cash_str = f"{cur}{r['繰越現金']:,.0f}"
        contrib_str = f"{cur}{r['拠出総額']:,.0f}"
        trades_str = f"{r['売買回数']}回"

        print(
            f"{rank:>2} | {r['手法名']:<12} | {shares_str:>10} | {avg_str:>14} | "
            f"{spent_str:>14} | {cash_str:>14} | {contrib_str:>14} | {trades_str}"
        )

    print("-" * 132)


# =========================
# メイン実験
# =========================

def run_practical_dva_experiment(
    tickers,
    start_date="2014-01-01",
    end_date="2023-12-31",
    total_budget_us=60000,
    total_budget_jp=6000000,
    score_threshold=0.55,
):
    print(f"--- 実用研究向け DVA 実験 ({start_date} ～ {end_date}) ---")
    print("DEBUG: PRACTICAL DVA SCORE VERSION")

    benchmark_vol_cache = {}

    for ticker in tickers:
        cfg = get_market_config(
            ticker,
            total_budget_us=total_budget_us,
            total_budget_jp=total_budget_jp
        )

        total_budget = cfg["total_budget"]
        benchmark = cfg["benchmark"]
        cur = cfg["currency_symbol"]
        lot_size = cfg["lot_size"]
        max_lots_per_trade = cfg["max_lots_per_trade"]

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']} / 単元: {lot_size}株】")

        # ベンチマーク取得
        if benchmark not in benchmark_vol_cache:
            bmk = yf.download(
                benchmark,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            bmk = normalize_df(bmk)

            if bmk.empty:
                print(f"  ベンチマーク取得失敗: {benchmark}")
                continue

            benchmark_vol_cache[benchmark] = calc_annualized_vol(bmk["Close"], window=20)

        mkt_vol_raw = benchmark_vol_cache[benchmark]

        # 銘柄取得
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        df = normalize_df(df)

        if df.empty:
            print("  データ取得失敗 or データなし")
            continue

        df = calc_indicators(df, mkt_vol_raw)

        monthly_budget, n_months = calc_monthly_budget_fixed_total(df, total_budget)
        contribution_schedule = build_monthly_contribution_schedule(df, total_budget)

        print(f"  対象月数: {n_months} か月")
        print(f"  総予算: {cur}{total_budget:,.0f}")
        print(f"  月予算: {cur}{monthly_budget:,.2f}")
        print(f"  DVAスコア閾値: {score_threshold}")

        # 実用DVA
        dva_result = execute_practical_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
            score_threshold=score_threshold,
            max_lots_per_trade=max_lots_per_trade,
        )

        # 比較用ベースライン
        first_result = execute_monthly_first(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
        )

        last_result = execute_monthly_last(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
        )

        results = [
            format_result_row("実用DVA", dva_result, cur),
            format_result_row("毎月(月初)", first_result, cur),
            format_result_row("毎月(月末)", last_result, cur),
        ]

        print_results_table(results)

        if not dva_result["trade_logs"].empty:
            print("  実用DVAの直近5件の約定:")
            recent = dva_result["trade_logs"].tail(5).copy()
            for _, row in recent.iterrows():
                print(
                    f"    {row['date'].date()} | "
                    f"価格 {cur}{row['price']:,.2f} | "
                    f"スコア {row['score']:.3f} | "
                    f"買付 {int(row['buy_shares'])}株 | "
                    f"約定 {cur}{row['cost']:,.0f}"
                )
        else:
            print("  実用DVAの約定はありません")

        print("  ※ 原則として『約定総額 + 繰越現金 = 拠出総額』となる")
        print("  ※ 実用DVAは、スコアが高くても単元資金が足りなければ買わない")


if __name__ == "__main__":
    run_practical_dva_experiment(
        tickers=["3080.T"],
        start_date="2007-01-01",
        end_date="2016-12-31",
        total_budget_us=60000,
        total_budget_jp=6000000,
        score_threshold=0.55,
    )
