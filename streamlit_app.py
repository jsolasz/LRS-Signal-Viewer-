#!/usr/bin/env python3
"""Streamlit dashboard for live futures trade state monitoring."""

from __future__ import annotations

import csv
import io
import importlib.util
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

try:
    import futures_trade_mapper as ftm
except ModuleNotFoundError:
    module_path = Path(__file__).resolve().with_name("futures_trade_mapper.py")
    if not module_path.exists():
        st.error(
            "Could not import `futures_trade_mapper` and file was not found next to "
            "`streamlit_app.py`. Ensure both files are committed to the repo root."
        )
        st.stop()
    spec = importlib.util.spec_from_file_location("futures_trade_mapper", module_path)
    if spec is None or spec.loader is None:
        st.error("Failed to load `futures_trade_mapper.py` from disk.")
        st.stop()
    ftm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftm)

try:
    from streamlit_autorefresh import st_autorefresh

    HAS_ST_AUTOREFRESH = True
except Exception:
    HAS_ST_AUTOREFRESH = False

ET = ZoneInfo("America/New_York")
SCRIPT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = SCRIPT_DIR / ".uploaded_trades"
UPLOAD_DIR.mkdir(exist_ok=True)
SIZE_FRACTIONS = [0.50, 0.25, 0.25]
CONTRACT_MULTIPLIERS = {
    "ES": 50.0,
    "NQ": 20.0,
    "YM": 5.0,
    "RTY": 50.0,
    "CL": 1000.0,
    "NG": 10000.0,
    "RB": 42000.0,
    "HO": 42000.0,
    "GC": 100.0,
    "SI": 5000.0,
    "HG": 25000.0,
    "PL": 50.0,
    "PA": 100.0,
    "ZB": 1000.0,
    "ZN": 1000.0,
    "ZF": 1000.0,
    "ZT": 2000.0,
    "KE": 5000.0,
    "ZW": 5000.0,
    "ZC": 5000.0,
    "ZS": 5000.0,
    "LE": 400.0,
    "HE": 400.0,
    "6E": 125000.0,
    "6B": 62500.0,
    "6J": 12500000.0,
    "6C": 100000.0,
    "6A": 100000.0,
    "6S": 125000.0,
}
ASSET_CLASS_BY_ROOT = {
    "ES": "Equity Index",
    "NQ": "Equity Index",
    "YM": "Equity Index",
    "RTY": "Equity Index",
    "CL": "Energy",
    "NG": "Energy",
    "RB": "Energy",
    "HO": "Energy",
    "GC": "Metals",
    "SI": "Metals",
    "HG": "Metals",
    "PL": "Metals",
    "PA": "Metals",
    "ZB": "Rates",
    "ZN": "Rates",
    "ZF": "Rates",
    "ZT": "Rates",
    "KE": "Ags",
    "ZW": "Ags",
    "ZC": "Ags",
    "ZS": "Ags",
    "LE": "Livestock",
    "HE": "Livestock",
    "6E": "FX",
    "6B": "FX",
    "6J": "FX",
    "6C": "FX",
    "6A": "FX",
    "6S": "FX",
}
MASTER_PASSWORD = "NewDay1574!"


def _require_password() -> None:
    if st.session_state.get("is_authenticated", False):
        return

    st.title("Protected App")
    st.caption("Enter password to access the dashboard.")
    pw = st.text_input("Password", type="password")
    if st.button("Unlock"):
        if pw == MASTER_PASSWORD:
            st.session_state["is_authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()


def _load_rows_from_bytes(file_name: str, raw_bytes: bytes) -> List[Dict[str, str]]:
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if len(lines) < 2:
        return []

    metadata = lines[0].strip()
    file_date = ftm.parse_file_date(metadata)
    reference_year = file_date.year if file_date else None
    signal_date = file_date.isoformat() if file_date else ""

    rows: List[Dict[str, str]] = []
    reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
    for idx, raw in enumerate(reader, start=3):
        row = ftm._clean_row(raw)
        contract_text = row.get("Future (System Direction)", "")
        if not contract_text:
            continue

        parsed = ftm.parse_contract(contract_text, reference_year)
        is_live, send_mult = ftm.parse_status(row.get("STATUS", ""))
        parsed_pos = ftm.parse_position(row.get("Position", ""))

        rows.append(
            {
                "source_file": file_name,
                "source_line": str(idx),
                "signal_date": signal_date,
                "contract_text": contract_text,
                "futures_clean": ftm.format_futures_clean(parsed),
                "contract_code": ftm.format_contract_code(parsed),
                "contract_name": ftm.format_contract_name(parsed),
                "root": parsed.root if parsed else "",
                "month_code": parsed.month_code if parsed else "",
                "year": str(parsed.year) if parsed else "",
                "side": ftm.parse_side(contract_text) or "",
                "entry": row.get("ENTRY", ""),
                "stop": row.get("STOP", ""),
                "target1": row.get("Target1", ""),
                "target2": row.get("Target2", ""),
                "target3": row.get("Target3", ""),
                "status": row.get("STATUS", ""),
                "is_live": str(is_live),
                "send_multiplier": "" if send_mult is None else str(send_mult),
                "close": row.get("CLOSE", ""),
                "position_raw": row.get("Position", ""),
                "position": "" if parsed_pos is None else str(parsed_pos),
                "account": row.get("Account", ""),
                "max_hold_date": row.get("Max Hold Date", ""),
            }
        )
    return rows


def _parse_price_with_root(value: object, root: object) -> float:
    parsed = ftm.parse_price(str(value), str(root or ""))
    return 0.0 if parsed is None else float(parsed)


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "entry_num" not in work.columns:
        work["entry_num"] = work.apply(lambda r: _parse_price_with_root(r.get("entry", ""), r.get("root", "")), axis=1)
    if "stop_num" not in work.columns:
        work["stop_num"] = work.apply(lambda r: _parse_price_with_root(r.get("stop", ""), r.get("root", "")), axis=1)
    if "target1_num" not in work.columns:
        work["target1_num"] = work.apply(lambda r: _parse_price_with_root(r.get("target1", ""), r.get("root", "")), axis=1)
    if "target2_num" not in work.columns:
        work["target2_num"] = work.apply(lambda r: _parse_price_with_root(r.get("target2", ""), r.get("root", "")), axis=1)
    if "target3_num" not in work.columns:
        work["target3_num"] = work.apply(lambda r: _parse_price_with_root(r.get("target3", ""), r.get("root", "")), axis=1)
    return work


def _save_uploaded_files(uploaded_files) -> List[Path]:
    saved_paths: List[Path] = []
    for uf in uploaded_files:
        path = UPLOAD_DIR / uf.name
        path.write_bytes(uf.getvalue())
        saved_paths.append(path)
    return saved_paths


def _load_rows_from_saved_files(paths: List[Path]) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for path in paths:
        try:
            all_rows.extend(_load_rows_from_bytes(path.name, path.read_bytes()))
        except Exception:
            continue
    return all_rows


def _state_bucket(row: Dict[str, str]) -> Tuple[str, str, str]:
    """Returns (state_key, state_label, hex_color)."""
    live = row.get("is_live") == "True"
    condition = row.get("condition", "")
    try:
        targets_hit = int(float(row.get("targets_hit", "0") or 0))
    except ValueError:
        targets_hit = 0

    if not live:
        return ("DORMANT", "Dormant", "#E5E7EB")

    if condition in {"history_unavailable", "history_skipped"}:
        return ("NO_DATA", "Live (no market data)", "#F3F4F6")

    if condition == "waiting_entry":
        return ("WAITING_ENTRY", "Live, waiting entry", "#FFF59D")

    if condition == "open_after_trigger":
        if targets_hit <= 0:
            return ("TRIGGERED", "Entry triggered, working T1 + stop", "#C8E6C9")
        if targets_hit == 1:
            return ("T1_OPEN", "Target1 hit, working T2 + stop@entry", "#81C784")
        if targets_hit == 2:
            return ("T2_OPEN", "Target2 hit, working T3 + stop@entry", "#388E3C")
        return ("T3_DONE", "All targets filled", "#1B5E20")

    if condition == "all_targets_filled":
        return ("T3_DONE", "All targets filled", "#1B5E20")

    if condition == "stopped_out":
        if targets_hit <= 0:
            return ("STOPPED", "Triggered, stopped out", "#EF4444")
        if targets_hit == 1:
            return ("T1_STOP", "Target1 hit, then stopped at entry", "#F59E0B")
        if targets_hit == 2:
            return ("T2_STOP", "Target2 hit, then stopped at entry", "#D97706")
        return ("STOPPED", "Stopped out", "#EF4444")

    if condition == "invalid_levels":
        return ("INVALID", "Invalid levels", "#FCA5A5")

    return ("UNKNOWN", condition or "Unknown", "#E5E7EB")


def _analyze(
    input_rows: List[Dict[str, str]],
    mode: str,
    intrabar_policy: str,
    live_only: bool,
    include_quotes: bool,
) -> pd.DataFrame:
    rows = ftm.filter_us_rows(input_rows)
    if live_only:
        rows = [r for r in rows if r.get("is_live") == "True"]

    quote_cache: Dict[str, ftm.QuoteResult] = {}
    history_cache: Dict[str, object] = {}
    output_rows: List[Dict[str, str]] = []

    for row in rows:
        out = dict(row)

        if include_quotes:
            quote = ftm.resolve_quote(row, mode, quote_cache)
        else:
            quote = ftm.QuoteResult(symbol=ftm.US_CONTINUOUS_SYMBOLS.get(row.get("root", ""), ""), price=None, timestamp=None, source="skipped", error=None)

        out["yahoo_symbol"] = quote.symbol
        out["quote_price"] = "" if quote.price is None else str(quote.price)
        out["quote_source"] = quote.source
        out["quote_error"] = quote.error or ""

        history_symbol, bars, history_error = _choose_history_live_window(row, mode, history_cache)
        if bars is None:
            state = ftm.TradeState(
                condition="history_unavailable",
                notes=history_error,
                working_stop_price=None,
                working_stop_size=0.0,
                next_target=None,
                next_target_price=None,
                remaining_size=0.0,
                targets_hit=0,
            )
            session_open = None
            market_price = None
        else:
            state = ftm.simulate_trade(row, bars, intrabar_policy)
            try:
                session_open = float(bars["Open"].iloc[0])
            except Exception:
                session_open = None
            try:
                market_price = float(bars["Close"].iloc[-1])
            except Exception:
                market_price = None

        out["history_symbol"] = history_symbol
        out["condition"] = state.condition
        out["working_stop_price"] = "" if state.working_stop_price is None else str(state.working_stop_price)
        out["working_stop_size"] = str(state.working_stop_size)
        out["next_target"] = state.next_target or ""
        out["next_target_price"] = "" if state.next_target_price is None else str(state.next_target_price)
        out["remaining_size"] = str(state.remaining_size)
        out["targets_hit"] = str(state.targets_hit)
        out["notes"] = state.notes
        out["session_open_price"] = "" if session_open is None else str(session_open)
        out["market_price"] = "" if market_price is None else str(market_price)

        state_key, state_label, state_color = _state_bucket(out)
        out["state_key"] = state_key
        out["state_label"] = state_label
        out["state_color"] = state_color

        output_rows.append(out)

    df = pd.DataFrame(output_rows)
    if df.empty:
        return df

    if "side" in df.columns:
        df["side"] = df["side"].map({"L": "BUY", "S": "SELL"}).fillna(df["side"])
    df = _ensure_price_columns(df)
    df["qty_abs"] = pd.to_numeric(df.get("position", 0), errors="coerce").abs().fillna(0.0)
    df["multiplier"] = df.get("root", "").map(CONTRACT_MULTIPLIERS).fillna(1.0).astype(float)
    df["asset_class"] = df.get("root", "").map(ASSET_CLASS_BY_ROOT).fillna("Other")
    is_buy = df.get("side", "") == "BUY"
    is_sell = df.get("side", "") == "SELL"
    df["max_risk"] = 0.0
    df.loc[is_buy, "max_risk"] = (
        (df.loc[is_buy, "entry_num"] - df.loc[is_buy, "stop_num"]).clip(lower=0.0)
        * df.loc[is_buy, "qty_abs"]
        * df.loc[is_buy, "multiplier"]
    )
    df.loc[is_sell, "max_risk"] = (
        (df.loc[is_sell, "stop_num"] - df.loc[is_sell, "entry_num"]).clip(lower=0.0)
        * df.loc[is_sell, "qty_abs"]
        * df.loc[is_sell, "multiplier"]
    )
    df["max_reward"] = 0.0
    df.loc[is_buy, "max_reward"] = (
        (
            0.50 * (df.loc[is_buy, "target1_num"] - df.loc[is_buy, "entry_num"])
            + 0.25 * (df.loc[is_buy, "target2_num"] - df.loc[is_buy, "entry_num"])
            + 0.25 * (df.loc[is_buy, "target3_num"] - df.loc[is_buy, "entry_num"])
        ).clip(lower=0.0)
        * df.loc[is_buy, "qty_abs"]
        * df.loc[is_buy, "multiplier"]
    )
    df.loc[is_sell, "max_reward"] = (
        (
            0.50 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target1_num"])
            + 0.25 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target2_num"])
            + 0.25 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target3_num"])
        ).clip(lower=0.0)
        * df.loc[is_sell, "qty_abs"]
        * df.loc[is_sell, "multiplier"]
    )
    df["max_entry_notional"] = df["entry_num"] * df["qty_abs"] * df["multiplier"]
    df["trade_id"] = (
        df["source_file"].astype(str)
        + "|"
        + df["source_line"].astype(str)
        + "|"
        + df["contract_code"].astype(str)
    )

    preferred = [
        "state_label",
        "contract_name",
        "contract_code",
        "side",
        "status",
        "position",
        "entry",
        "stop",
        "target1",
        "target2",
        "target3",
        "condition",
        "targets_hit",
        "max_risk",
        "max_reward",
        "working_stop_price",
        "working_stop_size",
        "next_target",
        "next_target_price",
        "history_symbol",
        "quote_price",
        "notes",
        "source_file",
        "source_line",
    ]
    available = [c for c in preferred if c in df.columns]
    return df[available + [c for c in df.columns if c not in available]]


def _render_outcome_scenarios(display_df: pd.DataFrame) -> None:
    st.subheader("Outcome Scenarios")
    if display_df.empty:
        st.info("No trades available for scenario analysis.")
        return

    scen = display_df.copy()
    # Backfill derived fields in case session cache holds an older dataframe shape.
    if "qty_abs" not in scen.columns:
        scen["qty_abs"] = pd.to_numeric(scen.get("position", 0), errors="coerce").abs().fillna(0.0)
    scen = _ensure_price_columns(scen)
    if "multiplier" not in scen.columns:
        scen["multiplier"] = scen.get("root", "").map(CONTRACT_MULTIPLIERS).fillna(1.0).astype(float)
    if "asset_class" not in scen.columns:
        scen["asset_class"] = scen.get("root", "").map(ASSET_CLASS_BY_ROOT).fillna("Other")
    if "max_entry_notional" not in scen.columns:
        scen["max_entry_notional"] = scen["entry_num"] * scen["qty_abs"] * scen["multiplier"]
    if "max_risk" not in scen.columns:
        is_buy = scen.get("side", "") == "BUY"
        is_sell = scen.get("side", "") == "SELL"
        scen["max_risk"] = 0.0
        scen.loc[is_buy, "max_risk"] = (
            (scen.loc[is_buy, "entry_num"] - scen.loc[is_buy, "stop_num"]).clip(lower=0.0)
            * scen.loc[is_buy, "qty_abs"]
            * scen.loc[is_buy, "multiplier"]
        )
        scen.loc[is_sell, "max_risk"] = (
            (scen.loc[is_sell, "stop_num"] - scen.loc[is_sell, "entry_num"]).clip(lower=0.0)
            * scen.loc[is_sell, "qty_abs"]
            * scen.loc[is_sell, "multiplier"]
        )
    if "max_reward" not in scen.columns:
        is_buy = scen.get("side", "") == "BUY"
        is_sell = scen.get("side", "") == "SELL"
        scen["max_reward"] = 0.0
        scen.loc[is_buy, "max_reward"] = (
            (
                0.50 * (scen.loc[is_buy, "target1_num"] - scen.loc[is_buy, "entry_num"])
                + 0.25 * (scen.loc[is_buy, "target2_num"] - scen.loc[is_buy, "entry_num"])
                + 0.25 * (scen.loc[is_buy, "target3_num"] - scen.loc[is_buy, "entry_num"])
            ).clip(lower=0.0)
            * scen.loc[is_buy, "qty_abs"]
            * scen.loc[is_buy, "multiplier"]
        )
        scen.loc[is_sell, "max_reward"] = (
            (
                0.50 * (scen.loc[is_sell, "entry_num"] - scen.loc[is_sell, "target1_num"])
                + 0.25 * (scen.loc[is_sell, "entry_num"] - scen.loc[is_sell, "target2_num"])
                + 0.25 * (scen.loc[is_sell, "entry_num"] - scen.loc[is_sell, "target3_num"])
            ).clip(lower=0.0)
            * scen.loc[is_sell, "qty_abs"]
            * scen.loc[is_sell, "multiplier"]
        )

    scen = scen[scen.get("is_live", "False") == "True"].copy()
    if scen.empty:
        st.info("No live trades for scenario analysis.")
        return

    long_mask = scen.get("side", "") == "BUY"
    short_mask = scen.get("side", "") == "SELL"

    max_possible_long_exposure = float(scen.loc[long_mask, "max_entry_notional"].sum())
    max_possible_short_exposure = float(scen.loc[short_mask, "max_entry_notional"].sum())
    max_risk_long = float(scen.loc[long_mask, "max_risk"].sum())
    max_risk_short = float(scen.loc[short_mask, "max_risk"].sum())
    max_risk_total = max_risk_long + max_risk_short
    max_reward_total = float(scen["max_reward"].sum())

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Max Long Exposure", _abbr_number(max_possible_long_exposure))
    a2.metric("Max Short Exposure", _abbr_number(max_possible_short_exposure))
    a3.metric("Max Risk Long", _abbr_number(max_risk_long))
    a4.metric("Max Risk Short", _abbr_number(max_risk_short))
    a5.metric("Max Risk Total", _abbr_number(max_risk_total))
    a6.metric("Max Reward Total", _abbr_number(max_reward_total))

    class_tbl = (
        scen.groupby("asset_class", as_index=False)
        .agg(
            max_risk=("max_risk", "sum"),
            max_reward=("max_reward", "sum"),
            gross_entry_notional=("max_entry_notional", "sum"),
            trades=("trade_id", "count"),
        )
        .sort_values("max_risk", ascending=False)
    )
    st.caption("Max Risk by Asset Class")
    st.dataframe(class_tbl, use_container_width=True, hide_index=True, height=230)


def _build_portfolio_snapshot(display_df: pd.DataFrame) -> pd.DataFrame:
    if display_df.empty:
        return pd.DataFrame()

    df = display_df.copy()
    # Backfill numeric/derived columns for stale cached dataframe shapes.
    if "qty_abs" not in df.columns:
        df["qty_abs"] = pd.to_numeric(df.get("position", 0), errors="coerce").abs().fillna(0.0)
    df = _ensure_price_columns(df)
    if "multiplier" not in df.columns:
        df["multiplier"] = df.get("root", "").map(CONTRACT_MULTIPLIERS).fillna(1.0).astype(float)
    if "max_risk" not in df.columns:
        is_buy = df.get("side", "") == "BUY"
        is_sell = df.get("side", "") == "SELL"
        df["max_risk"] = 0.0
        df.loc[is_buy, "max_risk"] = (
            (df.loc[is_buy, "entry_num"] - df.loc[is_buy, "stop_num"]).clip(lower=0.0)
            * df.loc[is_buy, "qty_abs"]
            * df.loc[is_buy, "multiplier"]
        )
        df.loc[is_sell, "max_risk"] = (
            (df.loc[is_sell, "stop_num"] - df.loc[is_sell, "entry_num"]).clip(lower=0.0)
            * df.loc[is_sell, "qty_abs"]
            * df.loc[is_sell, "multiplier"]
        )
    if "max_reward" not in df.columns:
        is_buy = df.get("side", "") == "BUY"
        is_sell = df.get("side", "") == "SELL"
        df["max_reward"] = 0.0
        df.loc[is_buy, "max_reward"] = (
            (
                0.50 * (df.loc[is_buy, "target1_num"] - df.loc[is_buy, "entry_num"])
                + 0.25 * (df.loc[is_buy, "target2_num"] - df.loc[is_buy, "entry_num"])
                + 0.25 * (df.loc[is_buy, "target3_num"] - df.loc[is_buy, "entry_num"])
            ).clip(lower=0.0)
            * df.loc[is_buy, "qty_abs"]
            * df.loc[is_buy, "multiplier"]
        )
        df.loc[is_sell, "max_reward"] = (
            (
                0.50 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target1_num"])
                + 0.25 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target2_num"])
                + 0.25 * (df.loc[is_sell, "entry_num"] - df.loc[is_sell, "target3_num"])
            ).clip(lower=0.0)
            * df.loc[is_sell, "qty_abs"]
            * df.loc[is_sell, "multiplier"]
        )

    triggered_conditions = {"open_after_trigger", "stopped_out", "all_targets_filled"}
    df = df[df["condition"].isin(triggered_conditions)].copy()
    if df.empty:
        return df

    df["qty"] = pd.to_numeric(df.get("position", 0), errors="coerce").abs().fillna(0.0)
    df["entry_px"] = pd.to_numeric(df.get("entry", 0), errors="coerce").fillna(0.0)
    df["session_open_px"] = pd.to_numeric(df.get("session_open_price", 0), errors="coerce")
    df["market_px"] = pd.to_numeric(df.get("market_price", 0), errors="coerce")
    missing_market = df["market_px"].isna() | (df["market_px"] == 0)
    df.loc[missing_market, "market_px"] = pd.to_numeric(df.get("quote_price", 0), errors="coerce")
    df["market_px"] = df["market_px"].fillna(df["entry_px"])
    df["session_open_px"] = df["session_open_px"].fillna(df["entry_px"])
    df["remaining_frac"] = pd.to_numeric(df.get("remaining_size", 0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["direction"] = df.get("side", "").map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    df["multiplier"] = (
        df.get("root", "")
        .map(CONTRACT_MULTIPLIERS)
        .fillna(1.0)
        .astype(float)
    )

    df["current_position"] = df["qty"] * df["remaining_frac"] * df["direction"]

    def _realized_row(r: pd.Series) -> float:
        qty = float(r.get("qty", 0.0))
        if qty <= 0:
            return 0.0
        entry = float(r.get("entry_px", 0.0))
        direction = float(r.get("direction", 0.0))
        mult = float(r.get("multiplier", 1.0))
        root = str(r.get("root", ""))
        if direction == 0:
            return 0.0
        try:
            targets_hit = int(float(r.get("targets_hit", 0)))
        except Exception:
            targets_hit = 0
        targets_hit = max(0, min(3, targets_hit))

        t_vals = [
            ftm.parse_price(str(r.get("target1", "")), root),
            ftm.parse_price(str(r.get("target2", "")), root),
            ftm.parse_price(str(r.get("target3", "")), root),
        ]
        fractions = SIZE_FRACTIONS

        realized = 0.0
        for i in range(targets_hit):
            if t_vals[i] is None:
                continue
            realized += qty * fractions[i] * direction * (float(t_vals[i]) - entry) * mult

        condition = str(r.get("condition", ""))
        if condition == "stopped_out":
            remaining = max(0.0, 1.0 - sum(fractions[:targets_hit]))
            if remaining > 0:
                stop_px = ftm.parse_price(str(r.get("stop", "")), root)
                if stop_px is None:
                    stop_px = entry
                if targets_hit >= 1:
                    stop_px = entry
                realized += qty * remaining * direction * (float(stop_px) - entry) * mult
        return realized

    df["realized_pl"] = df.apply(_realized_row, axis=1)
    df["unrealized_pl"] = df["current_position"] * (df["market_px"] - df["entry_px"]) * df["multiplier"]
    df["daily_pl"] = df["realized_pl"] + df["unrealized_pl"]
    base = (df["qty"].abs() * df["entry_px"] * df["multiplier"]).replace(0, pd.NA)
    df["daily_pl_pct"] = (df["daily_pl"] / base) * 100.0
    df["notional"] = (df["current_position"].abs() * df["market_px"] * df["multiplier"]).fillna(0.0)

    # Stop-based VaR for open positions only (with stop moved to entry after T1+).
    df["targets_hit_num"] = pd.to_numeric(df.get("targets_hit", 0), errors="coerce").fillna(0.0)
    df["effective_stop_px"] = df["stop_num"]
    df.loc[df["targets_hit_num"] >= 1, "effective_stop_px"] = df["entry_px"]
    open_mask = (df.get("condition", "") == "open_after_trigger") & (df["current_position"].abs() > 0)
    long_mask = open_mask & (df.get("side", "") == "BUY")
    short_mask = open_mask & (df.get("side", "") == "SELL")
    df["stop_var"] = 0.0
    df.loc[long_mask, "stop_var"] = (
        (df.loc[long_mask, "market_px"] - df.loc[long_mask, "effective_stop_px"]).clip(lower=0.0)
        * df.loc[long_mask, "current_position"].abs()
        * df.loc[long_mask, "multiplier"]
    )
    df.loc[short_mask, "stop_var"] = (
        (df.loc[short_mask, "effective_stop_px"] - df.loc[short_mask, "market_px"]).clip(lower=0.0)
        * df.loc[short_mask, "current_position"].abs()
        * df.loc[short_mask, "multiplier"]
    )

    return df


def _build_trade_action_rows(price_df: pd.DataFrame, row: Dict[str, str], x_col: str) -> List[Dict[str, object]]:
    side = str(row.get("side", "")).upper()
    root = str(row.get("root", ""))
    entry_value = ftm.parse_price(str(row.get("entry", "")), root)
    if entry_value is None or side not in {"BUY", "SELL", "L", "S"}:
        return []

    is_buy = side in {"BUY", "L"}
    trigger_idx = None
    for i in range(len(price_df)):
        hi = float(price_df.iloc[i]["High"])
        lo = float(price_df.iloc[i]["Low"])
        if (is_buy and hi >= float(entry_value)) or ((not is_buy) and lo <= float(entry_value)):
            trigger_idx = i
            break

    if trigger_idx is None:
        return []

    rows: List[Dict[str, object]] = [
        {
            "idx": trigger_idx,
            "ts": price_df.iloc[trigger_idx][x_col],
            "px": float(entry_value),
            "dir": "BUY" if is_buy else "SELL",
            "event": "Trigger",
            "remaining_after": 1.0,
        }
    ]

    try:
        targets_hit = int(float(str(row.get("targets_hit", "0") or "0")))
    except ValueError:
        targets_hit = 0
    targets_hit = max(0, min(3, targets_hit))
    target_values = [
        ftm.parse_price(str(row.get("target1", "")), root),
        ftm.parse_price(str(row.get("target2", "")), root),
        ftm.parse_price(str(row.get("target3", "")), root),
    ]
    target_action = "SELL" if is_buy else "BUY"

    remaining = 1.0
    last_event_idx = trigger_idx
    for t_idx in range(targets_hit):
        target_px = target_values[t_idx]
        if target_px is None:
            continue
        hit_idx = None
        for j in range(last_event_idx + 1, len(price_df)):
            hi = float(price_df.iloc[j]["High"])
            lo = float(price_df.iloc[j]["Low"])
            if is_buy and hi >= float(target_px):
                hit_idx = j
                break
            if (not is_buy) and lo <= float(target_px):
                hit_idx = j
                break
        if hit_idx is None:
            continue
        remaining = max(0.0, remaining - SIZE_FRACTIONS[t_idx])
        rows.append(
            {
                "idx": hit_idx,
                "ts": price_df.iloc[hit_idx][x_col],
                "px": float(target_px),
                "dir": target_action,
                "event": f"Target {t_idx + 1} Hit",
                "remaining_after": remaining,
            }
        )
        last_event_idx = hit_idx

    condition = str(row.get("condition", ""))
    if condition == "stopped_out":
        stop_price = ftm.parse_price(str(row.get("stop", "")), root)
        if targets_hit >= 1:
            stop_price = float(entry_value)

        if stop_price is not None:
            stop_idx = None
            for j in range(last_event_idx + 1, len(price_df)):
                hi = float(price_df.iloc[j]["High"])
                lo = float(price_df.iloc[j]["Low"])
                if is_buy and lo <= float(stop_price):
                    stop_idx = j
                    break
                if (not is_buy) and hi >= float(stop_price):
                    stop_idx = j
                    break
            if stop_idx is not None:
                rows.append(
                    {
                        "idx": stop_idx,
                        "ts": price_df.iloc[stop_idx][x_col],
                        "px": float(stop_price),
                        "dir": "SELL" if is_buy else "BUY",
                        "event": "Stop Out",
                        "remaining_after": 0.0,
                    }
                )

    return rows


def _session_exposure_stats(display_df: pd.DataFrame, mode: str) -> Dict[str, float]:
    if display_df.empty:
        return {"max": 0.0, "current": 0.0, "max_net": 0.0}

    history_cache: Dict[str, object] = {}
    events: List[Tuple[pd.Timestamp, float, float]] = []
    triggered_conditions = {"open_after_trigger", "stopped_out", "all_targets_filled"}

    for _, r in display_df.iterrows():
        row = r.to_dict()
        if row.get("condition") not in triggered_conditions:
            continue

        qty = abs(float(pd.to_numeric(row.get("position", 0), errors="coerce") or 0.0))
        entry = ftm.parse_price(str(row.get("entry", "")), str(row.get("root", "")))
        if qty <= 0 or entry is None:
            continue
        mult = float(CONTRACT_MULTIPLIERS.get(str(row.get("root", "")), 1.0))
        side_sign = 1.0 if str(row.get("side", "")).upper() == "BUY" else -1.0

        _, bars, _ = _choose_history_live_window(row, mode, history_cache)
        if bars is None or bars.empty:
            continue

        price_df = bars.reset_index().copy()
        x_col = price_df.columns[0]
        action_rows = _build_trade_action_rows(price_df, row, x_col)
        if not action_rows:
            continue

        notional_unit = qty * float(entry) * mult
        prev_remaining = 0.0
        for ev in action_rows:
            rem = float(ev.get("remaining_after", prev_remaining))
            gross_delta = notional_unit * (rem - prev_remaining)
            net_delta = side_sign * gross_delta
            events.append((pd.Timestamp(ev["ts"]), gross_delta, net_delta))
            prev_remaining = rem

    if not events:
        return {"max": 0.0, "current": 0.0, "max_net": 0.0}

    events.sort(key=lambda x: x[0])
    running_gross = 0.0
    running_net = 0.0
    max_exp = 0.0
    max_net = 0.0
    for _, gross_delta, net_delta in events:
        running_gross += gross_delta
        running_net += net_delta
        max_exp = max(max_exp, running_gross)
        max_net = max(max_net, abs(running_net))
    return {"max": max_exp, "current": running_gross, "max_net": max_net}


def _render_stat_cards(portfolio_df: pd.DataFrame, exposure_stats: Dict[str, float]) -> None:
    if portfolio_df.empty:
        st.info("No triggered trades for portfolio stats yet.")
        return

    realized_pl = float(portfolio_df["realized_pl"].sum())
    unrealized_pl = float(portfolio_df["unrealized_pl"].sum())
    daily_pl = float(portfolio_df["daily_pl"].sum())
    gross_exposure = float(portfolio_df["notional"].sum())
    max_exposure = float(exposure_stats.get("max", 0.0))
    max_net_exposure = float(exposure_stats.get("max_net", 0.0))
    stop_var_open = float(portfolio_df["stop_var"].sum()) if "stop_var" in portfolio_df.columns else 0.0

    valid_pct = portfolio_df["daily_pl_pct"].dropna()
    if valid_pct.empty:
        daily_pl_pct = 0.0
    else:
        daily_pl_pct = float(valid_pct.mean())

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Daily P/L", _abbr_number(daily_pl))
    c2.metric("Realized P/L", _abbr_number(realized_pl))
    c3.metric("Unrealized P/L", _abbr_number(unrealized_pl))
    c4.metric("Daily P/L %", f"{daily_pl_pct:,.2f}%")
    c5.metric("Gross Exposure (Now)", _abbr_number(gross_exposure))
    c6.metric("Max Exposure (Session)", _abbr_number(max_exposure))
    c7.metric("Max Net Exposure (Session)", _abbr_number(max_net_exposure))
    c8.metric("VaR to Stops (Open)", _abbr_number(stop_var_open))


def _abbr_number(value: float) -> str:
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    if x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.1f}K"
    return f"{sign}{x:,.2f}"


def _render_portfolio_table(portfolio_df: pd.DataFrame) -> None:
    st.subheader("Portfolio Snapshot (Triggered Trades)")
    if portfolio_df.empty:
        st.info("No triggered trades available.")
        return

    table_cols = [
        "source_file",
        "contract_code",
        "side",
        "state_label",
        "max_risk",
        "max_reward",
        "stop_var",
        "multiplier",
        "current_position",
        "entry_px",
        "market_px",
        "session_open_px",
        "realized_pl",
        "unrealized_pl",
        "daily_pl",
        "daily_pl_pct",
        "notes",
    ]
    show = portfolio_df[[c for c in table_cols if c in portfolio_df.columns]].copy()
    rename = {
        "source_file": "book",
        "contract_code": "contract",
        "state_label": "state",
        "max_risk": "max_risk",
        "max_reward": "max_reward",
        "stop_var": "var_to_stop",
        "multiplier": "multiplier",
        "current_position": "current_position",
        "entry_px": "entry",
        "market_px": "current_price",
        "session_open_px": "session_open",
        "realized_pl": "realized_pl",
        "unrealized_pl": "unrealized_pl",
        "daily_pl": "daily_pl",
        "daily_pl_pct": "daily_pl_%",
    }
    show = show.rename(columns=rename)
    st.dataframe(show, use_container_width=True, height=260, hide_index=True)


def _style_table(df: pd.DataFrame):
    colors = df.get("state_color", pd.Series(["#FFFFFF"] * len(df))).tolist()
    state_keys = df.get("state_key", pd.Series(["UNKNOWN"] * len(df))).tolist()

    dark_text_states = {
        "WAITING_ENTRY",
        "TRIGGERED",
        "T1_OPEN",
        "NO_DATA",
        "DORMANT",
        "UNKNOWN",
    }

    def color_row(row):
        bg = colors[row.name]
        state_key = state_keys[row.name]
        fg = "#111827" if state_key in dark_text_states else "#FFFFFF"
        style = f"background-color: {bg}; color: {fg}; font-weight: 600;"
        return [style] * len(row)

    return df.style.apply(color_row, axis=1).set_properties(**{"font-size": "14px"})


def _table_label(source_file: str) -> str:
    lower = source_file.lower()
    if "global_macro" in lower or "global macro" in lower:
        return "Global Macro"
    if "eqint" in lower:
        return "IntradayPlus"
    return source_file


def _render_section(df_section: pd.DataFrame, section_title: str) -> None:
    st.subheader(section_title)
    counts = df_section["state_label"].value_counts().reset_index()
    counts.columns = ["State", "Count"]
    c1, c2 = st.columns([4, 1.3])
    with c1:
        st.dataframe(_style_table(df_section), use_container_width=True, height=420)
    with c2:
        st.dataframe(counts, use_container_width=True, height=220)


def _render_single_selector(display_df: pd.DataFrame) -> None:
    st.subheader("Trade Selection")
    selector_df = display_df.copy()
    selector_df["source_group"] = selector_df["source_file"].astype(str).apply(_table_label)
    selector_cols = [
        "trade_id",
        "source_group",
        "contract_code",
        "side",
        "state_label",
        "condition",
        "targets_hit",
        "max_risk",
        "max_reward",
        "source_line",
    ]
    cols = [c for c in selector_cols if c in selector_df.columns]
    selected = st.dataframe(
        selector_df[cols],
        use_container_width=True,
        height=260,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tbl_all_select",
        column_config={"trade_id": None},
    )
    if selected and selected.selection and selected.selection.rows:
        selected_idx = selected.selection.rows[0]
        st.session_state["selected_trade_id"] = selector_df.iloc[selected_idx]["trade_id"]


def _df_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "EMPTY"
    cols = [c for c in ["trade_id", "contract_code", "status", "condition", "targets_hit", "working_stop_price", "next_target", "notes", "quote_price"] if c in df.columns]
    snapshot = df[cols].sort_values(by=[c for c in ["contract_code", "status", "condition"] if c in cols]).reset_index(drop=True)
    return json.dumps(snapshot.astype(str).to_dict(orient="records"), sort_keys=True)


def _session_window_et() -> Tuple[datetime, datetime]:
    now_et = datetime.now(ET)
    yesterday_et = (now_et - timedelta(days=1)).date()
    start_et = datetime.combine(yesterday_et, time(hour=18, minute=0), tzinfo=ET)
    return start_et, now_et


def _choose_history_live_window(row: Dict[str, str], mode: str, history_cache: Dict[str, object]):
    start_et, end_et = _session_window_et()
    last_error = "No history candidates"

    for symbol in ftm.resolve_history_symbol(row, mode):
        cache_key = f"{symbol}|{start_et.isoformat()}|{end_et.isoformat()}"
        if cache_key not in history_cache:
            try:
                # Pull a slightly wider window and then hard-filter to ET session bounds.
                hist = ftm.yf.Ticker(symbol).history(
                    start=(start_et - timedelta(hours=12)),
                    end=(end_et + timedelta(hours=1)),
                    interval="5m",
                    prepost=True,
                    auto_adjust=False,
                )
                if hist is not None and not hist.empty:
                    idx = pd.to_datetime(hist.index)
                    # Normalize index to ET for consistent window filtering.
                    if getattr(idx, "tz", None) is None:
                        idx = idx.tz_localize("UTC").tz_convert(ET)
                    else:
                        idx = idx.tz_convert(ET)
                    hist.index = idx
                    hist = hist[(hist.index >= start_et) & (hist.index <= end_et)]
                history_cache[cache_key] = hist if hist is not None and not hist.empty else None
            except Exception as exc:  # pragma: no cover
                history_cache[cache_key] = None
                last_error = str(exc)

        hist = history_cache[cache_key]
        if hist is not None and {"High", "Low", "Open", "Close"}.issubset(set(hist.columns)):
            return symbol, hist, ""
        last_error = f"No intraday bars for {symbol}"

    return "", None, last_error


def _build_price_levels_chart(bars: pd.DataFrame, row: Dict[str, str]):
    price_df = bars.reset_index().copy()
    x_col = price_df.columns[0]
    price_min = float(price_df["Low"].min())
    price_max = float(price_df["High"].max())
    line = (
        alt.Chart(price_df)
        .mark_line(color="#60A5FA")
        .encode(
            x=alt.X(f"{x_col}:T", title="Time"),
            y=alt.Y("Close:Q", title="Price"),
            tooltip=[
                alt.Tooltip(f"{x_col}:T"),
                alt.Tooltip("Open:Q"),
                alt.Tooltip("High:Q"),
                alt.Tooltip("Low:Q"),
                alt.Tooltip("Close:Q"),
            ],
        )
    )

    level_specs = [
        ("entry", row.get("entry", ""), "#FDE047", "Entry"),
        ("stop", row.get("stop", ""), "#EF4444", "Stop"),
        ("target1", row.get("target1", ""), "#86EFAC", "Target 1"),
        ("target2", row.get("target2", ""), "#4ADE80", "Target 2"),
        ("target3", row.get("target3", ""), "#15803D", "Target 3"),
    ]

    level_rows = []
    for _, raw, color, label in level_specs:
        value = ftm.parse_price(str(raw), str(row.get("root", "")))
        if value is not None:
            level_rows.append({"level": float(value), "label": label, "color": color})
            price_min = min(price_min, float(value))
            price_max = max(price_max, float(value))

    span = max(price_max - price_min, max(abs(price_max), 1.0) * 0.0025)
    pad = span * 0.12
    y_scale = alt.Scale(domain=[price_min - pad, price_max + pad], zero=False, nice=False)

    marker_layer = None
    side = str(row.get("side", "")).upper()
    root = str(row.get("root", ""))
    entry_value = ftm.parse_price(str(row.get("entry", "")), root)
    if entry_value is not None and side in {"BUY", "SELL", "L", "S"}:
        is_buy = side in {"BUY", "L"}
        trigger_idx = None
        for i in range(len(price_df)):
            hi = float(price_df.iloc[i]["High"])
            lo = float(price_df.iloc[i]["Low"])
            if (is_buy and hi >= float(entry_value)) or ((not is_buy) and lo <= float(entry_value)):
                trigger_idx = i
                break

        if trigger_idx is not None:
            marker_rows = [
                {
                    "ts": price_df.iloc[trigger_idx][x_col],
                    "px": float(entry_value),
                    "dir": "BUY" if is_buy else "SELL",
                    "event": "Trigger",
                }
            ]

            # Add target action markers (long targets = SELL, short targets = BUY).
            try:
                targets_hit = int(float(str(row.get("targets_hit", "0") or "0")))
            except ValueError:
                targets_hit = 0
            target_values = [
                ftm.parse_price(str(row.get("target1", "")), root),
                ftm.parse_price(str(row.get("target2", "")), root),
                ftm.parse_price(str(row.get("target3", "")), root),
            ]
            target_action = "SELL" if is_buy else "BUY"
            last_event_idx = trigger_idx
            for t_idx in range(min(targets_hit, 3)):
                target_px = target_values[t_idx]
                if target_px is None:
                    continue
                hit_idx = None
                # Enforce event ordering: targets are searched after the prior event.
                for j in range(last_event_idx + 1, len(price_df)):
                    hi = float(price_df.iloc[j]["High"])
                    lo = float(price_df.iloc[j]["Low"])
                    if is_buy and hi >= float(target_px):
                        hit_idx = j
                        break
                    if (not is_buy) and lo <= float(target_px):
                        hit_idx = j
                        break
                if hit_idx is None:
                    continue
                marker_rows.append(
                    {
                        "ts": price_df.iloc[hit_idx][x_col],
                        "px": float(target_px),
                        "dir": target_action,
                        "event": f"Target {t_idx + 1} Hit",
                    }
                )
                last_event_idx = hit_idx

            # Add stop-out action marker (e.g. short stopped out => BUY stop marker).
            condition = str(row.get("condition", ""))
            if condition == "stopped_out":
                stop_price = ftm.parse_price(str(row.get("stop", "")), root)
                if targets_hit >= 1:
                    stop_price = float(entry_value)

                if stop_price is not None:
                    stop_idx = None
                    # Stop action must occur after the latest trigger/target event.
                    for j in range(last_event_idx + 1, len(price_df)):
                        hi = float(price_df.iloc[j]["High"])
                        lo = float(price_df.iloc[j]["Low"])
                        if is_buy and lo <= float(stop_price):
                            stop_idx = j
                            break
                        if (not is_buy) and hi >= float(stop_price):
                            stop_idx = j
                            break

                    if stop_idx is not None:
                        marker_rows.append(
                            {
                                "ts": price_df.iloc[stop_idx][x_col],
                                "px": float(stop_price),
                                "dir": "SELL" if is_buy else "BUY",
                                "event": "Stop Out",
                            }
                        )

            marker_df = pd.DataFrame(marker_rows)
            buy_df = marker_df[marker_df["dir"] == "BUY"]
            sell_df = marker_df[marker_df["dir"] == "SELL"]

            layers = []
            if not buy_df.empty:
                layers.append(
                    alt.Chart(buy_df)
                    .mark_point(shape="triangle-up", size=260, filled=True, color="#2563EB")
                    .encode(
                        x=alt.X("ts:T"),
                        y=alt.Y("px:Q", scale=y_scale),
                        tooltip=[alt.Tooltip("event:N"), alt.Tooltip("dir:N"), alt.Tooltip("px:Q"), alt.Tooltip("ts:T")],
                    )
                )
            if not sell_df.empty:
                layers.append(
                    alt.Chart(sell_df)
                    .mark_point(shape="triangle-down", size=260, filled=True, color="#DC2626")
                    .encode(
                        x=alt.X("ts:T"),
                        y=alt.Y("px:Q", scale=y_scale),
                        tooltip=[alt.Tooltip("event:N"), alt.Tooltip("dir:N"), alt.Tooltip("px:Q"), alt.Tooltip("ts:T")],
                    )
                )
            if layers:
                marker_layer = layers[0]
                for extra in layers[1:]:
                    marker_layer = marker_layer + extra

    if not level_rows:
        chart = line.encode(y=alt.Y("Close:Q", title="Price", scale=y_scale))
        if marker_layer is not None:
            chart = chart + marker_layer
        return chart.properties(height=430)

    levels_df = pd.DataFrame(level_rows)
    rules = (
        alt.Chart(levels_df)
        .mark_rule(strokeDash=[8, 4], size=2)
        .encode(
            y=alt.Y("level:Q", scale=y_scale),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=levels_df["label"].tolist(), range=levels_df["color"].tolist()),
                legend=alt.Legend(title="Trade Levels"),
            ),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("level:Q")],
        )
    )

    chart = line.encode(y=alt.Y("Close:Q", title="Price", scale=y_scale)) + rules
    if marker_layer is not None:
        chart = chart + marker_layer
    return chart.properties(height=430)


def main() -> None:
    st.set_page_config(page_title="Futures Trade Monitor", layout="wide")
    _require_password()
    st.title("Futures Trade Monitor")
    st.caption("Live table of trade states with entry/target/stop progression")
    start_et, end_et = _session_window_et()
    st.caption(f"Evaluation window: {start_et.strftime('%Y-%m-%d %I:%M %p ET')} to {end_et.strftime('%Y-%m-%d %I:%M %p ET')}")

    with st.sidebar:
        st.header("Controls")
        mode = st.selectbox("Yahoo mapping mode", ["both", "contract", "continuous"], index=0)
        intrabar_policy = st.selectbox("Intrabar priority", ["stop-first", "target-first"], index=0)
        live_only = st.checkbox("Live only (Send x)", value=True)
        include_quotes = st.checkbox("Include latest quote", value=False)
        auto_refresh = st.checkbox("Enable auto refresh", value=False)
        refresh_seconds = st.slider("Refresh seconds", min_value=5, max_value=120, value=20, step=5)
        manual_refresh = st.button("Refresh now")
        clear_files = st.button("Clear uploaded files")

    if manual_refresh:
        st.rerun()

    uploaded_files = st.file_uploader(
        "Upload trade CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more trade sheets (e.g., eqint and Global_Macro).",
    )

    if clear_files:
        for p in UPLOAD_DIR.glob("*.csv"):
            p.unlink(missing_ok=True)
        st.success("Cleared saved uploaded files.")

    if uploaded_files:
        saved_paths = _save_uploaded_files(uploaded_files)
    else:
        saved_paths = sorted(UPLOAD_DIR.glob("*.csv"))

    if not saved_paths:
        st.info("Upload your trade CSV files to start the live table.")
        return

    st.caption("Using uploaded files: " + ", ".join(p.name for p in saved_paths))
    input_rows = _load_rows_from_saved_files(saved_paths)

    if auto_refresh and HAS_ST_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="trade_table_refresh")
    elif auto_refresh and not HAS_ST_AUTOREFRESH:
        st.info("Install `streamlit-autorefresh` for smooth in-app refresh (no page blink): `pip install streamlit-autorefresh`")

    df = _analyze(
        input_rows=input_rows,
        mode=mode,
        intrabar_policy=intrabar_policy,
        live_only=live_only,
        include_quotes=include_quotes,
    )

    if df.empty:
        st.warning("No trades found in uploaded files (or no US contracts matched).")
        return

    current_sig = _df_signature(df)
    if "last_df_sig" not in st.session_state:
        st.session_state.last_df_sig = current_sig
        st.session_state.last_df = df.copy()
    elif current_sig != st.session_state.last_df_sig:
        st.session_state.last_df_sig = current_sig
        st.session_state.last_df = df.copy()

    display_df = st.session_state.last_df
    if "trade_id" not in display_df.columns:
        display_df = display_df.copy()
        display_df["trade_id"] = (
            display_df.get("source_file", pd.Series([""] * len(display_df))).astype(str)
            + "|"
            + display_df.get("source_line", pd.Series([""] * len(display_df))).astype(str)
            + "|"
            + display_df.get("contract_code", pd.Series([""] * len(display_df))).astype(str)
        )
        st.session_state.last_df = display_df

    portfolio_df = _build_portfolio_snapshot(display_df.reset_index(drop=True))
    exposure_stats = _session_exposure_stats(display_df.reset_index(drop=True), mode)
    tab_live, tab_risk = st.tabs(["Live Monitor", "Pre-Trade Risk"])

    with tab_live:
        st.subheader("Portfolio Stats")
        _render_stat_cards(portfolio_df, exposure_stats)
        _render_portfolio_table(portfolio_df)

        if "source_file" not in display_df.columns:
            _render_section(display_df.reset_index(drop=True), "All Trades")
        else:
            for source_file, section in display_df.groupby("source_file", sort=True):
                _render_section(section.reset_index(drop=True), _table_label(str(source_file)))

        st.subheader("Color Legend")
        legend = pd.DataFrame(
            [
                {"State": "Live, waiting entry", "Color": "Yellow"},
                {"State": "Entry triggered, working T1 + stop", "Color": "Light Green"},
                {"State": "Target1 hit, working T2 + stop@entry", "Color": "Medium Green"},
                {"State": "Target2 hit, working T3 + stop@entry", "Color": "Dark Green"},
                {"State": "All targets filled", "Color": "Darkest Green"},
                {"State": "Triggered, stopped out", "Color": "Red"},
                {"State": "Target1/2 hit, then stopped at entry", "Color": "Orange"},
                {"State": "Live (no market data)", "Color": "Light Gray"},
            ]
        )
        st.dataframe(legend, use_container_width=True, height=260)

        _render_single_selector(display_df.reset_index(drop=True))
        st.subheader("Chart Inspector")
        selected_trade_id = st.session_state.get("selected_trade_id")
        if not selected_trade_id or selected_trade_id not in set(display_df["trade_id"].tolist()):
            selected_trade_id = display_df.iloc[0]["trade_id"]
            st.session_state["selected_trade_id"] = selected_trade_id

        selected_row = display_df.loc[display_df["trade_id"] == selected_trade_id].iloc[0].to_dict()
        st.caption(
            f"Selected: {selected_row.get('source_file', '')} | {selected_row.get('contract_code', '')} | "
            f"{selected_row.get('side', '')} | line {selected_row.get('source_line', '')}"
        )

        history_symbol, bars, history_error = _choose_history_live_window(selected_row, mode, {})
        if bars is None:
            st.warning(f"Could not load intraday bars: {history_error}")
        else:
            st.caption(f"Symbol: {history_symbol}")
            try:
                loaded_start = pd.to_datetime(bars.index.min()).tz_convert(ET)
                loaded_end = pd.to_datetime(bars.index.max()).tz_convert(ET)
                st.caption(
                    "Bars loaded: "
                    f"{loaded_start.strftime('%Y-%m-%d %I:%M %p ET')} to "
                    f"{loaded_end.strftime('%Y-%m-%d %I:%M %p ET')}"
                )
            except Exception:
                pass
            st.altair_chart(_build_price_levels_chart(bars, selected_row), use_container_width=True)

        detail_cols = [
            "state_label",
            "condition",
            "notes",
            "contract_name",
            "contract_code",
            "side",
            "entry",
            "stop",
            "target1",
            "target2",
            "target3",
            "working_stop_price",
            "next_target",
            "next_target_price",
            "targets_hit",
        ]
        details = {k: selected_row.get(k, "") for k in detail_cols}
        st.json(details)

    with tab_risk:
        _render_outcome_scenarios(display_df.reset_index(drop=True))


if __name__ == "__main__":
    main()
