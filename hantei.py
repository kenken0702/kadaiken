#!/usr/bin/env python3
"""
株式強気・弱気・中立判定システム
==============================
Yahoo Finance (yfinance) を利用し、2016-01-01 〜 2025-12-31 の過去10年間の
個別株データを取得し、様々な基準で総合的に「強気 / 弱気 / 中立」を判定する。

判定基準:
  1. 長期トレンド (200日SMAとの位置関係)
  2. 短期トレンド (50日SMAとの位置関係)
  3. ゴールデン/デッドクロス (50日SMA vs 200日SMA)
  4. MACD (移動平均収束拡散)
  5. RSI (相対力指数)
  6. ボリンジャーバンド (%B)
  7. 10年間年率リターン
  8. 価格モメンタム (過去6ヶ月・12ヶ月)
  9. 52週ハイ/ローからの位置
 10. 出来高トレンド

各基準を -1(弱気) / 0(中立) / +1(強気) でスコア化し、
合計スコアから総合判定を行う。
"""

import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
START_DATE = "2007-01-01"
END_DATE   = "2016-12-31"


# ═══════════════════════════════════════════════
# データ取得
# ═══════════════════════════════════════════════
def fetch_stock_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Yahoo Finance から日足データを取得する（堅牢なエラーハンドリング付き）"""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"データ取得失敗: {ticker}（ティッカーが正しいか確認してください）")
        # マルチインデックス対応（yfinance のバージョン差異吸収）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # 必須列チェック
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"必須列 '{col}' が存在しません: {ticker}")
        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 250:
            raise ValueError(f"データ不足: {len(df)}行（最低250行必要）")
        return df
    except Exception as e:
        print(f"[ERROR] データ取得エラー ({ticker}): {e}")
        return None


# ═══════════════════════════════════════════════
# テクニカル指標の計算
# ═══════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """各種テクニカル指標を計算して DataFrame に追加する"""
    close = df["Close"]
    volume = df["Volume"]

    # --- 移動平均線 ---
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()
    df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df["MACD"]       = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # --- RSI (14日) ---
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # --- ボリンジャーバンド (20日, 2σ) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid * 100
    df["BB_PctB"]  = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    # --- モメンタム (過去N日リターン) ---
    df["Momentum_6M"]  = close.pct_change(126)   # ≈6ヶ月
    df["Momentum_12M"] = close.pct_change(252)   # ≈12ヶ月

    # --- 52週ハイ/ロー ---
    rolling_high = close.rolling(252, min_periods=126).max()
    rolling_low  = close.rolling(252, min_periods=126).min()
    df["Pct_from_52High"] = (close - rolling_high) / rolling_high * 100
    df["Pct_from_52Low"]  = (close - rolling_low)  / rolling_low  * 100

    # --- 出来高トレンド (20日平均出来高 vs 直近5日平均) ---
    vol_20 = volume.rolling(20).mean()
    vol_5  = volume.rolling(5).mean()
    df["Volume_Ratio"] = vol_5 / vol_20.replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════
# 各判定基準（シグナル生成）
#   戻り値: (score, label, detail)
#   score: +1 = 強気, 0 = 中立, -1 = 弱気
# ═══════════════════════════════════════════════
def check_long_term_trend(df: pd.DataFrame):
    """1. 長期トレンド: 終値が200日SMAより上なら強気"""
    latest = df.iloc[-1]
    if pd.isna(latest["SMA_200"]):
        return 0, "中立", "200日SMAデータ不足"
    if latest["Close"] > latest["SMA_200"] * 1.02:
        return 1, "強気", f"終値 ${latest['Close']:.2f} > 200日SMA ${latest['SMA_200']:.2f} (+2%以上)"
    elif latest["Close"] < latest["SMA_200"] * 0.98:
        return -1, "弱気", f"終値 ${latest['Close']:.2f} < 200日SMA ${latest['SMA_200']:.2f} (-2%以上)"
    else:
        return 0, "中立", f"終値 ${latest['Close']:.2f} ≒ 200日SMA ${latest['SMA_200']:.2f} (±2%以内)"


def check_short_term_trend(df: pd.DataFrame):
    """2. 短期トレンド: 終値が50日SMAより上なら強気"""
    latest = df.iloc[-1]
    if pd.isna(latest["SMA_50"]):
        return 0, "中立", "50日SMAデータ不足"
    if latest["Close"] > latest["SMA_50"]:
        return 1, "強気", f"終値 ${latest['Close']:.2f} > 50日SMA ${latest['SMA_50']:.2f}"
    else:
        return -1, "弱気", f"終値 ${latest['Close']:.2f} < 50日SMA ${latest['SMA_50']:.2f}"


def check_golden_dead_cross(df: pd.DataFrame):
    """3. ゴールデン/デッドクロス: 50日SMA と 200日SMA の位置関係"""
    latest = df.iloc[-1]
    if pd.isna(latest["SMA_50"]) or pd.isna(latest["SMA_200"]):
        return 0, "中立", "SMAデータ不足"
    if latest["SMA_50"] > latest["SMA_200"]:
        return 1, "強気", f"50日SMA ${latest['SMA_50']:.2f} > 200日SMA ${latest['SMA_200']:.2f} (ゴールデンクロス状態)"
    else:
        return -1, "弱気", f"50日SMA ${latest['SMA_50']:.2f} < 200日SMA ${latest['SMA_200']:.2f} (デッドクロス状態)"


def check_macd(df: pd.DataFrame):
    """4. MACD: MACDラインがシグナルラインより上なら強気"""
    latest = df.iloc[-1]
    if pd.isna(latest["MACD"]) or pd.isna(latest["MACD_Signal"]):
        return 0, "中立", "MACDデータ不足"
    if latest["MACD"] > latest["MACD_Signal"] and latest["MACD"] > 0:
        return 1, "強気", f"MACD {latest['MACD']:.4f} > シグナル {latest['MACD_Signal']:.4f} かつMACD正"
    elif latest["MACD"] < latest["MACD_Signal"] and latest["MACD"] < 0:
        return -1, "弱気", f"MACD {latest['MACD']:.4f} < シグナル {latest['MACD_Signal']:.4f} かつMACD負"
    else:
        return 0, "中立", f"MACD {latest['MACD']:.4f} / シグナル {latest['MACD_Signal']:.4f} (シグナル不一致)"


def check_rsi(df: pd.DataFrame):
    """5. RSI: 70超過=強気(買われすぎ警戒)、30未満=弱気(売られすぎ)、30-70=中立"""
    latest = df.iloc[-1]
    rsi = latest["RSI_14"]
    if pd.isna(rsi):
        return 0, "中立", "RSIデータ不足"
    if rsi > 70:
        return 1, "強気", f"RSI {rsi:.1f} > 70 (買われすぎ圏内だが強い上昇圧力)"
    elif rsi < 30:
        return -1, "弱気", f"RSI {rsi:.1f} < 30 (売られすぎ圏内、下落圧力)"
    elif rsi >= 55:
        return 1, "強気", f"RSI {rsi:.1f} (55-70: 上昇トレンド寄り)"
    elif rsi <= 45:
        return -1, "弱気", f"RSI {rsi:.1f} (30-45: 下落トレンド寄り)"
    else:
        return 0, "中立", f"RSI {rsi:.1f} (45-55: 中立帯)"


def check_bollinger(df: pd.DataFrame):
    """6. ボリンジャーバンド %B: 1.0超=強気、0未満=弱気"""
    latest = df.iloc[-1]
    pctb = latest["BB_PctB"]
    if pd.isna(pctb):
        return 0, "中立", "BB %Bデータ不足"
    if pctb > 0.8:
        return 1, "強気", f"BB %B {pctb:.2f} (バンド上方に位置、強い上昇圧力)"
    elif pctb < 0.2:
        return -1, "弱気", f"BB %B {pctb:.2f} (バンド下方に位置、強い下落圧力)"
    else:
        return 0, "中立", f"BB %B {pctb:.2f} (バンド中央付近)"


def check_annualized_return(df: pd.DataFrame):
    """7. 10年間年率リターン: 10%超=強気、0%未満=弱気"""
    start_price = df["Close"].iloc[0]
    end_price   = df["Close"].iloc[-1]
    years       = len(df) / 252.0
    if years <= 0 or start_price <= 0:
        return 0, "中立", "期間データ不足"
    cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
    if cagr > 10:
        return 1, "強気", f"年率リターン(CAGR) {cagr:+.1f}% (>10%: 優秀)"
    elif cagr < 0:
        return -1, "弱気", f"年率リターン(CAGR) {cagr:+.1f}% (<0%: マイナス成長)"
    elif cagr >= 5:
        return 1, "強気", f"年率リターン(CAGR) {cagr:+.1f}% (5-10%: 健全)"
    else:
        return 0, "中立", f"年率リターン(CAGR) {cagr:+.1f}% (0-5%: 低成長)"


def check_momentum(df: pd.DataFrame):
    """8. 価格モメンタム: 6ヶ月・12ヶ月リターンから総合判定"""
    m6  = df["Momentum_6M"].iloc[-1]
    m12 = df["Momentum_12M"].iloc[-1]
    if pd.isna(m6) or pd.isna(m12):
        return 0, "中立", "モメンタムデータ不足"
    m6_pct  = m6  * 100
    m12_pct = m12 * 100
    if m6_pct > 10 and m12_pct > 15:
        return 1, "強気", f"6M {m6_pct:+.1f}% / 12M {m12_pct:+.1f}% (両方強い上昇モメンタム)"
    elif m6_pct < -10 and m12_pct < -15:
        return -1, "弱気", f"6M {m6_pct:+.1f}% / 12M {m12_pct:+.1f}% (両方強い下落モメンタム)"
    elif m6_pct > 0 and m12_pct > 0:
        return 1, "強気", f"6M {m6_pct:+.1f}% / 12M {m12_pct:+.1f}% (プラスモメンタム)"
    elif m6_pct < 0 and m12_pct < 0:
        return -1, "弱気", f"6M {m6_pct:+.1f}% / 12M {m12_pct:+.1f}% (マイナスモメンタム)"
    else:
        return 0, "中立", f"6M {m6_pct:+.1f}% / 12M {m12_pct:+.1f}% (方向不一致)"


def check_52week_position(df: pd.DataFrame):
    """9. 52週ハイ/ローからの位置: ハイに近い=強気、ローに近い=弱気"""
    pct_high = df["Pct_from_52High"].iloc[-1]
    pct_low  = df["Pct_from_52Low"].iloc[-1]
    if pd.isna(pct_high) or pd.isna(pct_low):
        return 0, "中立", "52週データ不足"
    if pct_high > -5:
        return 1, "強気", f"52週ハイから {pct_high:.1f}% (ハイ圏内、強気トレンド)"
    elif pct_low < 5:
        return -1, "弱気", f"52週ローから {pct_low:.1f}% (ローの近く、弱気トレンド)"
    else:
        return 0, "中立", f"52週ハイから {pct_high:.1f}% / ローから {pct_low:.1f}% (中間帯)"


def check_volume_trend(df: pd.DataFrame):
    """10. 出来高トレンド: 直近出来高が平均より多い+価格上昇=強気"""
    latest = df.iloc[-1]
    vol_ratio = latest["Volume_Ratio"]
    if pd.isna(vol_ratio):
        return 0, "中立", "出来高データ不足"
    close_5_ago = df["Close"].iloc[-6] if len(df) >= 6 else df["Close"].iloc[0]
    price_chg = (latest["Close"] - close_5_ago) / close_5_ago * 100
    if vol_ratio > 1.3 and price_chg > 0:
        return 1, "強気", f"出来高比 {vol_ratio:.2f}x + 価格上昇 {price_chg:+.1f}% (出来高伴い上昇)"
    elif vol_ratio > 1.3 and price_chg < 0:
        return -1, "弱気", f"出来高比 {vol_ratio:.2f}x + 価格下落 {price_chg:+.1f}% (出来高伴い下落)"
    elif vol_ratio < 0.7:
        return 0, "中立", f"出来高比 {vol_ratio:.2f}x (低調、トレンド弱し)"
    else:
        return 0, "中立", f"出来高比 {vol_ratio:.2f}x + 価格変動 {price_chg:+.1f}% (通常範囲)"


# ═══════════════════════════════════════════════
# 総合判定エンジン
# ═══════════════════════════════════════════════
def analyze_stock(ticker: str) -> dict:
    """全判定基準を実行し、総合判定結果を返す"""
    print(f"\n{'='*70}")
    print(f"  📊 分析対象: {ticker}  (期間: {START_DATE} 〜 {END_DATE})")
    print(f"{'='*70}")

    # データ取得
    df = fetch_stock_data(ticker)
    if df is None:
        return {"ticker": ticker, "error": "データ取得失敗"}

    # 指標計算
    df = compute_indicators(df)

    # 銘柄情報
    info = {}
    try:
        yf_ticker = yf.Ticker(ticker)
        info["name"]     = yf_ticker.info.get("shortName", ticker)
        info["sector"]   = yf_ticker.info.get("sector", "N/A")
        info["currency"] = yf_ticker.info.get("currency", "USD")
        info["exchange"] = yf_ticker.info.get("exchange", "N/A")
    except Exception:
        info = {"name": ticker, "sector": "N/A", "currency": "USD", "exchange": "N/A"}

    # 各判定基準を実行
    checks = [
        ("1. 長期トレンド (200日SMA)",    check_long_term_trend),
        ("2. 短期トレンド (50日SMA)",      check_short_term_trend),
        ("3. ゴールデン/デッドクロス",     check_golden_dead_cross),
        ("4. MACD",                       check_macd),
        ("5. RSI (14日)",                 check_rsi),
        ("6. ボリンジャーバンド (%B)",     check_bollinger),
        ("7. 10年年率リターン (CAGR)",     check_annualized_return),
        ("8. 価格モメンタム (6M/12M)",     check_momentum),
        ("9. 52週ハイ/ロー位置",           check_52week_position),
        ("10. 出来高トレンド",             check_volume_trend),
    ]

    results = []
    total_score = 0

    for name, func in checks:
        try:
            score, label, detail = func(df)
        except Exception as e:
            score, label, detail = 0, "中立", f"判定エラー: {e}"
        results.append({
            "criterion": name,
            "score": score,
            "label": label,
            "detail": detail,
        })
        total_score += score

    # 総合判定
    max_score = len(checks)  # 10
    normalized = total_score / max_score  # -1.0 〜 +1.0

    if total_score >= 4:
        overall = "🟢 強気 (BULLISH)"
    elif total_score <= -4:
        overall = "🔴 弱気 (BEARISH)"
    else:
        overall = "🟡 中立 (NEUTRAL)"

    # 信頼度 (スコアの絶対値が大きいほど高い)
    confidence = min(abs(total_score) / max_score * 100, 100)

    # 結果表示
    print(f"\n  銘柄名: {info.get('name', ticker)}")
    print(f"  セクター: {info.get('sector', 'N/A')}")
    print(f"  取引所: {info.get('exchange', 'N/A')}")
    print(f"  データ期間: {df.index[0].date()} 〜 {df.index[-1].date()} ({len(df)}日)")
    print(f"\n  {'判定基準':<28} {'判定':<12} {'スコア':>5}  詳細")
    print(f"  {'-'*90}")

    for r in results:
        icon = "🟢" if r["score"] > 0 else ("🔴" if r["score"] < 0 else "🟡")
        print(f"  {icon} {r['criterion']:<26} {r['label']:<12} {r['score']:>+5}  {r['detail']}")

    print(f"  {'-'*90}")
    print(f"  {'合計スコア':<28} {'':12} {total_score:>+5} / {max_score}")
    print(f"  {'正規化スコア':<28} {'':12} {normalized:>+.2f}")
    print(f"  {'信頼度':<28} {'':12} {confidence:>.0f}%")
    print(f"\n  ═══ 総合判定: {overall} ═══  (信頼度 {confidence:.0f}%)")
    print(f"{'='*70}\n")

    return {
        "ticker": ticker,
        "info": info,
        "data_days": len(df),
        "results": results,
        "total_score": total_score,
        "max_score": max_score,
        "normalized": normalized,
        "overall": overall,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════
# 複数銘柄一括分析
# ═══════════════════════════════════════════════
def analyze_multiple_stocks(tickers: list) -> list:
    """複数銘柄を一括分析し、サマリーテーブルを表示する"""
    all_results = []
    for t in tickers:
        result = analyze_stock(t)
        all_results.append(result)

    # サマリーテーブル
    print(f"\n{'='*70}")
    print(f"  📋 複数銘柄サマリー ({len(tickers)}銘柄)")
    print(f"{'='*70}")
    print(f"  {'ティッカー':<10} {'銘柄名':<25} {'スコア':>6} {'判定':<20} {'信頼度':>6}")
    print(f"  {'-'*70}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['ticker']:<10} {'(エラー)':<25} {'N/A':>6} {'N/A':<20} {'N/A':>6}")
            continue
        name_short = r["info"].get("name", r["ticker"])[:23]
        print(f"  {r['ticker']:<10} {name_short:<25} {r['total_score']:>+6} {r['overall']:<20} {r['confidence']:>5.0f}%")

    print(f"  {'-'*70}\n")
    return all_results


# ═══════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    # ── テスト実行 ──
    # 単一銘柄
    print("=" * 70)
    print("  🐂🐻 株式強気・弱気・中立判定システム")
    print("  期間: 2016-01-01 〜 2025-12-31 (過去10年)")
    print("=" * 70)

    # 例: Apple (AAPL) を分析
    analyze_stock("AAPL")

    # 複数銘柄の一括分析例（コメントアウトを外して使用）
    # analyze_multiple_stocks(["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
