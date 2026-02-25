#!/usr/bin/env python3
"""Streamlit dashboard for live futures trade state monitoring."""

from __future__ import annotations

import csv
import io
import importlib.util
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
OPEN_CACHE_PATH = UPLOAD_DIR / ".open_positions_cache.csv"
SIZE_FRACTIONS = [0.50, 0.25, 0.25]
CONTRACT_MULTIPLIERS = {
    "ES": 50.0,
    "NQ": 20.0,
    "MNQ": 2.0,
    "YM": 5.0,
    "NKD": 5.0,
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
    "MNQ": "Equity Index",
    "YM": "Equity Index",
    "NKD": "Equity Index",
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
RAW_ROW_COLUMNS = [
    "source_file",
    "source_line",
    "signal_date",
    "contract_text",
    "program",
    "futures_clean",
    "contract_code",
    "contract_name",
    "root",
    "month_code",
    "year",
    "side",
    "entry",
    "stop",
    "target1",
    "target2",
    "target3",
    "status",
    "is_live",
    "send_multiplier",
    "close",
    "position_raw",
    "position",
    "account",
    "max_hold_date",
    "cache_carry",
    "cache_targets_hit",
    "cache_remaining_size",
]


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
        contract_text = row.get("Future (System Direction)", "") or row.get("FUTURE", "")
        if not contract_text:
            continue

        parsed = ftm.parse_contract(contract_text, reference_year)
        # Ignore footer/open-position/non-signal rows.
        if parsed is None:
            continue
        if ftm.parse_float(row.get("ENTRY", "")) is None:
            continue
        if ftm.parse_float(row.get("STOP", "")) is None:
            continue
        if ftm.parse_float(row.get("Target1", "")) is None:
            continue
        if ftm.parse_float(row.get("Target2", "")) is None:
            continue
        if ftm.parse_float(row.get("Target3", "")) is None:
            continue
        is_live, send_mult = ftm.parse_status(row.get("STATUS", ""))
        parsed_pos = ftm.parse_position(row.get("Position", ""))
        parse_program_fn = getattr(ftm, "parse_program", None)
        program = parse_program_fn(contract_text) if callable(parse_program_fn) else ""

        rows.append(
            {
                "source_file": file_name,
                "source_line": str(idx),
                "signal_date": signal_date,
                "contract_text": contract_text,
                "program": program,
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


def _strategy_group(source_file: object) -> str:
    return _table_label(str(source_file or ""))


def _position_key(row: Dict[str, object]) -> str:
    side_raw = str(row.get("side", "")).upper()
    side_norm = {"BUY": "L", "SELL": "S"}.get(side_raw, side_raw)
    parts = [
        _strategy_group(row.get("source_file", "")),
        str(row.get("account", "")),
        str(row.get("contract_code", "")),
        side_norm,
        str(row.get("entry", "")),
        str(row.get("stop", "")),
        str(row.get("target1", "")),
        str(row.get("target2", "")),
        str(row.get("target3", "")),
    ]
    return "|".join(parts)


def _load_open_positions_cache() -> List[Dict[str, str]]:
    if not OPEN_CACHE_PATH.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with OPEN_CACHE_PATH.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                clean = {k: (v or "") for k, v in raw.items()}
                # Backward-compatible defaults for legacy cache files.
                if not clean.get("cache_carry", "").strip():
                    clean["cache_carry"] = "1"
                if not clean.get("cache_targets_hit", "").strip():
                    clean["cache_targets_hit"] = "0"
                if not clean.get("cache_remaining_size", "").strip():
                    pos = abs(float(pd.to_numeric(clean.get("position", 0), errors="coerce") or 0.0))
                    clean["cache_remaining_size"] = "1.0" if pos > 0 else "0.0"
                rows.append(clean)
    except Exception:
        return []
    return rows


def _merge_with_open_cache(input_rows: List[Dict[str, str]], cached_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in cached_rows:
        merged[_position_key(row)] = row
    for row in input_rows:
        key = _position_key(row)
        existing = merged.get(key, {})
        combined = dict(row)
        existing_carry = str(existing.get("cache_carry", "")).strip() == "1"
        input_is_live = str(row.get("is_live", "")).strip() == "True"
        input_pos_abs = abs(float(pd.to_numeric(row.get("position", 0), errors="coerce") or 0.0))

        # If we already have a carried open position and today's row is inactive/blank,
        # keep the cached open row as the authoritative state.
        if existing_carry and (not input_is_live or input_pos_abs <= 0):
            merged[key] = existing
            continue

        for k in ["cache_carry", "cache_targets_hit", "cache_remaining_size"]:
            if existing.get(k, "") and not combined.get(k, ""):
                combined[k] = existing.get(k, "")
        merged[key] = combined
    return list(merged.values())


def _save_open_positions_cache(display_df: pd.DataFrame) -> None:
    if display_df.empty:
        # Preserve existing cache if this run has no snapshot rows.
        return
    keep = display_df[
        (display_df.get("condition", "") == "open_after_trigger")
        & (pd.to_numeric(display_df.get("current_position", 0), errors="coerce").abs() > 0)
    ].copy()
    if keep.empty:
        # Preserve existing cache to avoid accidental wipe on temporary/no-data runs.
        return

    out_rows: List[Dict[str, str]] = []
    for _, r in keep.iterrows():
        row_out = {col: str(r.get(col, "")) for col in RAW_ROW_COLUMNS}
        qty_abs = abs(float(pd.to_numeric(r.get("qty", r.get("position", 0)), errors="coerce") or 0.0))
        cur_abs = abs(float(pd.to_numeric(r.get("current_position", 0), errors="coerce") or 0.0))
        remaining = 0.0 if qty_abs <= 0 else max(0.0, min(1.0, cur_abs / qty_abs))
        row_out["cache_carry"] = "1"
        row_out["cache_targets_hit"] = str(int(float(pd.to_numeric(r.get("targets_hit", 0), errors="coerce") or 0)))
        row_out["cache_remaining_size"] = str(remaining)
        out_rows.append(row_out)

    dedup: Dict[str, Dict[str, str]] = {}
    for row in out_rows:
        dedup[_position_key(row)] = row

    with OPEN_CACHE_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(list(dedup.values()))


def _write_open_positions_cache_rows(rows: List[Dict[str, str]]) -> None:
    if not rows:
        OPEN_CACHE_PATH.write_text("", encoding="utf-8")
        return
    dedup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        normalized = {col: str(row.get(col, "")) for col in RAW_ROW_COLUMNS}
        dedup[_position_key(normalized)] = normalized
    with OPEN_CACHE_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(list(dedup.values()))


def _parse_max_hold_cutoff_et(max_hold_date_text: object) -> Optional[datetime]:
    text = str(max_hold_date_text or "").strip()
    if not text:
        return None
    if " " in text:
        text = text.split(" ", 1)[0].strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
        try:
            d = datetime.strptime(text, fmt).date()
            return datetime.combine(d, time(hour=16, minute=0), tzinfo=ET)
        except ValueError:
            continue
    return None


def _close_price_at_or_before_cutoff(
    bars: Optional[pd.DataFrame], cutoff_et: datetime, fallback_price: Optional[float]
) -> Optional[float]:
    if bars is None or bars.empty:
        return fallback_price
    try:
        idx = pd.to_datetime(bars.index)
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize(ET)
        else:
            idx = idx.tz_convert(ET)
        cutoff_bars = bars.loc[idx <= cutoff_et]
        if cutoff_bars.empty:
            return fallback_price
        return float(cutoff_bars["Close"].iloc[-1])
    except Exception:
        return fallback_price


def _parse_price_with_root(value: object, root: object) -> Optional[float]:
    text = str(value)
    root_text = str(root or "").upper()

    # Prefer mapper's root-aware parser when available.
    if hasattr(ftm, "parse_price"):
        parsed = ftm.parse_price(text, root_text)
        return None if parsed is None else float(parsed)

    # Backward-compat fallback: handle ZN/ZB 32nds sheet format locally.
    if root_text in {"ZN", "ZB"}:
        import re

        match = re.fullmatch(r"\s*([+-]?\d+)\.(\d{3})\s*", text)
        if match:
            whole_text, frac_text = match.groups()
            try:
                whole = int(whole_text)
                frac_32 = int(frac_text) / 10.0
                sign = -1.0 if whole < 0 else 1.0
                return float(whole) + sign * (frac_32 / 32.0)
            except ValueError:
                pass

    # Final fallback for older mapper versions.
    parsed = ftm.parse_float(text)
    return None if parsed is None else float(parsed)


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "entry_num" not in work.columns:
        work["entry_num"] = (
            work.apply(lambda r: _parse_price_with_root(r.get("entry", ""), r.get("root", "")), axis=1)
            .fillna(0.0)
            .astype(float)
        )
    if "stop_num" not in work.columns:
        work["stop_num"] = (
            work.apply(lambda r: _parse_price_with_root(r.get("stop", ""), r.get("root", "")), axis=1)
            .fillna(0.0)
            .astype(float)
        )
    if "target1_num" not in work.columns:
        work["target1_num"] = (
            work.apply(lambda r: _parse_price_with_root(r.get("target1", ""), r.get("root", "")), axis=1)
            .fillna(0.0)
            .astype(float)
        )
    if "target2_num" not in work.columns:
        work["target2_num"] = (
            work.apply(lambda r: _parse_price_with_root(r.get("target2", ""), r.get("root", "")), axis=1)
            .fillna(0.0)
            .astype(float)
        )
    if "target3_num" not in work.columns:
        work["target3_num"] = (
            work.apply(lambda r: _parse_price_with_root(r.get("target3", ""), r.get("root", "")), axis=1)
            .fillna(0.0)
            .astype(float)
        )
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
        return ("NO_DATA", "Live (no market data)", "#E5E7EB")

    if condition == "waiting_entry":
        return ("WAITING_ENTRY", "Live, waiting entry", "#FFF59D")

    if condition == "open_after_trigger":
        if targets_hit <= 0:
            return ("TRIGGERED", "Entry Triggered, working T1 + stop", "#F59E0B")
        if targets_hit == 1:
            return ("T1_OPEN", "Target1 Filled, working T2 + stop@entry", "#93C5FD")
        if targets_hit == 2:
            return ("T2_OPEN", "Target2 Filled, working T3 + stop@entry", "#60A5FA")
        return ("T3_DONE", "All targets Filled", "#15803D")

    if condition == "all_targets_filled":
        return ("T3_DONE", "All targets Filled", "#15803D")

    if condition == "max_hold_closed":
        outcome = str(row.get("max_hold_outcome", "flat")).lower()
        if outcome == "positive":
            return ("HOLD_CLOSED", "Max hold close (profit/flat/loss)", "#22C55E")
        if outcome == "negative":
            return ("HOLD_CLOSED", "Max hold close (profit/flat/loss)", "#EF4444")
        return ("HOLD_CLOSED", "Max hold close (profit/flat/loss)", "#9CA3AF")

    if condition == "stopped_out":
        if targets_hit <= 0:
            return ("STOPPED", "Hard Stop", "#EF4444")
        if targets_hit == 1:
            return ("T1_STOP", "Target 1 Filled, Stop @ Entry", "#22C55E")
        if targets_hit == 2:
            return ("T2_STOP", "Target 2 Filled, Stop @ Target 1", "#16A34A")
        return ("STOPPED", "Stopped out", "#EF4444")

    if condition == "invalid_levels":
        return ("INVALID", "Invalid levels", "#FCA5A5")

    return ("UNKNOWN", condition or "Unknown", "#E5E7EB")


def _ensure_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    needed = {"state_key", "state_label", "state_color"}
    if needed.issubset(set(work.columns)):
        missing_mask = (
            work["state_key"].astype(str).str.strip().eq("")
            | work["state_label"].astype(str).str.strip().eq("")
            | work["state_color"].astype(str).str.strip().eq("")
        )
        if not missing_mask.any():
            return work
    rows = []
    for _, r in work.iterrows():
        row = {k: r.get(k, "") for k in work.columns}
        state_key, state_label, state_color = _state_bucket(row)
        row["state_key"] = state_key
        row["state_label"] = state_label
        row["state_color"] = state_color
        rows.append(row)
    return pd.DataFrame(rows)


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
    now_et = datetime.now(ET)

    for row in rows:
        out = dict(row)
        out["max_hold_outcome"] = ""

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

        # Carry previously open positions across daily uploads.
        # If a cached open position does not retrigger within the current session window,
        # keep it open with the cached progression state.
        cache_carry = str(row.get("cache_carry", "")).strip() == "1"
        if cache_carry and state.condition in {"waiting_entry", "history_unavailable", "history_skipped"}:
            root = str(row.get("root", ""))
            entry_px = _parse_price_with_root(row.get("entry", ""), root)
            stop_px = _parse_price_with_root(row.get("stop", ""), root)
            t1_px = _parse_price_with_root(row.get("target1", ""), root)
            t2_px = _parse_price_with_root(row.get("target2", ""), root)
            t3_px = _parse_price_with_root(row.get("target3", ""), root)
            try:
                cached_targets = int(float(str(row.get("cache_targets_hit", "0") or "0")))
            except Exception:
                cached_targets = 0
            cached_targets = max(0, min(3, cached_targets))
            try:
                cached_remaining = float(str(row.get("cache_remaining_size", "0") or "0"))
            except Exception:
                cached_remaining = 0.0
            cached_remaining = max(0.0, min(1.0, cached_remaining))
            next_target_name = ""
            next_target_price = None
            if cached_targets == 0:
                next_target_name, next_target_price = "target1", t1_px
            elif cached_targets == 1:
                next_target_name, next_target_price = "target2", t2_px
            elif cached_targets == 2:
                next_target_name, next_target_price = "target3", t3_px
            working_stop = None
            if cached_remaining > 0:
                if cached_targets >= 1 and entry_px is not None:
                    working_stop = float(entry_px)
                elif stop_px is not None:
                    working_stop = float(stop_px)
            state = ftm.TradeState(
                condition="open_after_trigger",
                notes="carried open position from prior session cache",
                working_stop_price=working_stop,
                working_stop_size=cached_remaining,
                next_target=next_target_name if cached_remaining > 0 and next_target_price is not None else None,
                next_target_price=None if next_target_price is None else float(next_target_price),
                remaining_size=cached_remaining,
                targets_hit=cached_targets,
            )

        cutoff_et = _parse_max_hold_cutoff_et(row.get("max_hold_date", ""))
        if cutoff_et is not None and now_et >= cutoff_et and state.condition == "open_after_trigger":
            forced_exit_px = _close_price_at_or_before_cutoff(bars, cutoff_et, market_price)
            if forced_exit_px is not None:
                market_price = float(forced_exit_px)
            try:
                entry_px = ftm.parse_price(str(row.get("entry", "")), str(row.get("root", "")))
            except Exception:
                entry_px = None
            side_raw = str(row.get("side", "")).upper()
            is_buy_side = side_raw in {"BUY", "L"}
            if forced_exit_px is not None and entry_px is not None and side_raw in {"BUY", "SELL", "L", "S"}:
                pnl_points = float(forced_exit_px) - float(entry_px)
                signed = pnl_points if is_buy_side else -pnl_points
                if signed > 1e-12:
                    out["max_hold_outcome"] = "positive"
                elif signed < -1e-12:
                    out["max_hold_outcome"] = "negative"
                else:
                    out["max_hold_outcome"] = "flat"
            else:
                out["max_hold_outcome"] = "flat"
            prior_notes = state.notes or ""
            forced_note = f"Max hold close executed at 4:00 PM ET on {cutoff_et.strftime('%Y-%m-%d')}."
            if forced_exit_px is not None:
                forced_note = f"{forced_note} Exit px {forced_exit_px:.6g}."
            merged_notes = f"{prior_notes} | {forced_note}" if prior_notes else forced_note
            state = ftm.TradeState(
                condition="max_hold_closed",
                notes=merged_notes,
                working_stop_price=None,
                working_stop_size=0.0,
                next_target=None,
                next_target_price=None,
                remaining_size=0.0,
                targets_hit=state.targets_hit,
            )

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
        "program",
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

    triggered_conditions = {"open_after_trigger", "stopped_out", "all_targets_filled", "max_hold_closed"}
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
            _parse_price_with_root(r.get("target1", ""), root),
            _parse_price_with_root(r.get("target2", ""), root),
            _parse_price_with_root(r.get("target3", ""), root),
        ]
        fractions = SIZE_FRACTIONS

        realized = 0.0
        for i in range(targets_hit):
            if t_vals[i] is None:
                continue
            realized += qty * fractions[i] * direction * (float(t_vals[i]) - entry) * mult

        condition = str(r.get("condition", ""))
        if condition in {"stopped_out", "max_hold_closed"}:
            remaining = max(0.0, 1.0 - sum(fractions[:targets_hit]))
            if remaining > 0:
                exit_px = None
                if condition == "max_hold_closed":
                    try:
                        exit_px = float(r.get("market_px", entry))
                    except Exception:
                        exit_px = entry
                else:
                    exit_px = _parse_price_with_root(r.get("stop", ""), root)
                    if exit_px is None:
                        exit_px = entry
                    if targets_hit >= 1:
                        exit_px = entry
                realized += qty * remaining * direction * (float(exit_px) - entry) * mult
        return realized

    df["realized_pl"] = df.apply(_realized_row, axis=1)
    df["unrealized_pl"] = df["current_position"] * (df["market_px"] - df["entry_px"]) * df["multiplier"]
    df["daily_pl"] = df["realized_pl"] + df["unrealized_pl"]
    base = (df["qty"].abs() * df["entry_px"] * df["multiplier"]).replace(0, pd.NA)
    df["daily_pl_pct"] = (df["daily_pl"] / base) * 100.0
    df["unrealized_pl_pct"] = (df["unrealized_pl"] / base) * 100.0
    df["realized_pl_pct"] = (df["realized_pl"] / base) * 100.0
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
    entry_value = _parse_price_with_root(row.get("entry", ""), root)
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
        _parse_price_with_root(row.get("target1", ""), root),
        _parse_price_with_root(row.get("target2", ""), root),
        _parse_price_with_root(row.get("target3", ""), root),
    ]
    target_action = "SELL" if is_buy else "BUY"

    remaining = 1.0
    last_event_idx = trigger_idx
    for t_idx in range(targets_hit):
        target_px = target_values[t_idx]
        if target_px is None:
            continue
        hit_idx = None
        # Allow same-bar fills after trigger/previous target.
        for j in range(last_event_idx, len(price_df)):
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
        stop_price = _parse_price_with_root(row.get("stop", ""), root)
        if targets_hit >= 1:
            stop_price = float(entry_value)

        if stop_price is not None:
            stop_idx = None
            # Stop may occur on the same bar as the latest target event.
            for j in range(last_event_idx, len(price_df)):
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
    elif condition == "max_hold_closed":
        close_price = _parse_price_with_root(row.get("market_price", ""), root)
        if close_price is None:
            close_price = _parse_price_with_root(row.get("quote_price", ""), root)
        if close_price is None:
            close_price = float(price_df.iloc[-1]["Close"])
        close_idx = len(price_df) - 1
        cutoff_et = _parse_max_hold_cutoff_et(row.get("max_hold_date", ""))
        if cutoff_et is not None:
            for j in range(last_event_idx + 1, len(price_df)):
                ts = pd.Timestamp(price_df.iloc[j][x_col])
                if ts.tzinfo is None:
                    ts = ts.tz_localize(ET)
                else:
                    ts = ts.tz_convert(ET)
                if ts >= cutoff_et:
                    close_idx = j
                    break
        rows.append(
            {
                "idx": close_idx,
                "ts": price_df.iloc[close_idx][x_col],
                "px": float(close_price),
                "dir": "SELL" if is_buy else "BUY",
                "event": "Max Hold Close",
                "remaining_after": 0.0,
            }
        )

    return rows


def _session_exposure_stats(display_df: pd.DataFrame, mode: str) -> Dict[str, float]:
    if display_df.empty:
        return {"max": 0.0, "current": 0.0, "max_net": 0.0}

    history_cache: Dict[str, object] = {}
    events: List[Tuple[pd.Timestamp, float, float]] = []
    triggered_conditions = {"open_after_trigger", "stopped_out", "all_targets_filled", "max_hold_closed"}

    for _, r in display_df.iterrows():
        row = r.to_dict()
        if row.get("condition") not in triggered_conditions:
            continue

        qty = abs(float(pd.to_numeric(row.get("position", 0), errors="coerce") or 0.0))
        entry = _parse_price_with_root(row.get("entry", ""), row.get("root", ""))
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


def _value_color(value: float) -> str:
    if value > 0:
        return "#22C55E"
    if value < 0:
        return "#EF4444"
    return "#9CA3AF"


def _metric_card(label: str, value_text: str, color: str, subtitle: str = "") -> str:
    subtitle_html = (
        f'<div style="font-size:11px;color:#6B7280;margin-top:4px;">{subtitle}</div>'
        if subtitle
        else ""
    )
    return f"""
    <div style="
        border:1px solid #1f2937;
        border-radius:10px;
        padding:12px 10px;
        background:#0b1220;
        min-height:84px;
    ">
      <div style="font-size:12px;color:#9CA3AF;margin-bottom:6px;">{label}</div>
      <div style="font-size:22px;font-weight:700;color:{color};line-height:1.1;">{value_text}</div>
      {subtitle_html}
    </div>
    """


def _render_stat_cards(portfolio_df: pd.DataFrame, exposure_stats: Dict[str, float]) -> None:
    if portfolio_df.empty:
        st.info("No triggered trades for portfolio stats yet.")
        return

    realized_pl = float(portfolio_df["realized_pl"].sum())
    unrealized_pl = float(portfolio_df["unrealized_pl"].sum())
    daily_pl = float(portfolio_df["daily_pl"].sum())
    gross_exposure_now = float(portfolio_df["notional"].sum())
    max_exposure = float(exposure_stats.get("max", 0.0))
    max_net_exposure = float(exposure_stats.get("max_net", 0.0))
    stop_var_open = float(portfolio_df["stop_var"].sum()) if "stop_var" in portfolio_df.columns else 0.0

    base = float((portfolio_df["qty"].abs() * portfolio_df["entry_px"] * portfolio_df["multiplier"]).sum())
    daily_pl_pct = (daily_pl / base * 100.0) if base > 0 else 0.0
    unrealized_pl_pct = (unrealized_pl / base * 100.0) if base > 0 else 0.0
    realized_pl_pct = (realized_pl / base * 100.0) if base > 0 else 0.0

    market_value = float((portfolio_df["current_position"] * portfolio_df["market_px"] * portfolio_df["multiplier"]).sum())
    long_value = float(
        (portfolio_df.loc[portfolio_df["current_position"] > 0, "current_position"]
         * portfolio_df.loc[portfolio_df["current_position"] > 0, "market_px"]
         * portfolio_df.loc[portfolio_df["current_position"] > 0, "multiplier"]).sum()
    )
    short_value = float(
        (portfolio_df.loc[portfolio_df["current_position"] < 0, "current_position"].abs()
         * portfolio_df.loc[portfolio_df["current_position"] < 0, "market_px"]
         * portfolio_df.loc[portfolio_df["current_position"] < 0, "multiplier"]).sum()
    )

    r1 = st.columns(6)
    r1[0].markdown(_metric_card("Total P/L USD", _abbr_number(daily_pl), _value_color(daily_pl)), unsafe_allow_html=True)
    r1[1].markdown(_metric_card("Total P/L %", f"{daily_pl_pct:,.2f}%", _value_color(daily_pl_pct)), unsafe_allow_html=True)
    r1[2].markdown(_metric_card("Unrealized P/L USD", _abbr_number(unrealized_pl), _value_color(unrealized_pl)), unsafe_allow_html=True)
    r1[3].markdown(_metric_card("Unrealized P/L %", f"{unrealized_pl_pct:,.2f}%", _value_color(unrealized_pl_pct)), unsafe_allow_html=True)
    r1[4].markdown(_metric_card("Realized P/L USD", _abbr_number(realized_pl), _value_color(realized_pl)), unsafe_allow_html=True)
    r1[5].markdown(_metric_card("Realized P/L %", f"{realized_pl_pct:,.2f}%", _value_color(realized_pl_pct)), unsafe_allow_html=True)

    r2 = st.columns(5)
    r2[0].markdown(_metric_card("Market Value", _abbr_number(market_value), _value_color(market_value)), unsafe_allow_html=True)
    r2[1].markdown(_metric_card("Long Value", _abbr_number(long_value), "#22C55E"), unsafe_allow_html=True)
    r2[2].markdown(_metric_card("Short Value", _abbr_number(short_value), "#EF4444"), unsafe_allow_html=True)
    r2[3].markdown(
        _metric_card(
            "Gross Exposure",
            f"{_abbr_number(gross_exposure_now)} / {_abbr_number(max_exposure)}",
            "#60A5FA",
            "Current / Max (session)",
        ),
        unsafe_allow_html=True,
    )
    r2[4].markdown(_metric_card("VaR to Stop", _abbr_number(stop_var_open), "#F59E0B"), unsafe_allow_html=True)


def _render_program_stats(portfolio_df: pd.DataFrame) -> None:
    st.subheader("Performance by Program")
    if portfolio_df.empty:
        st.info("No triggered trades for program stats yet.")
        return

    work = portfolio_df.copy()
    work["program_group"] = work.get("source_file", "").astype(str).apply(_table_label)
    for col in ["daily_pl", "realized_pl", "unrealized_pl", "stop_var", "notional"]:
        if col not in work.columns:
            work[col] = 0.0

    grouped = (
        work.groupby("program_group", as_index=False)
        .agg(
            daily_pl=("daily_pl", "sum"),
            realized_pl=("realized_pl", "sum"),
            unrealized_pl=("unrealized_pl", "sum"),
            var_to_stop=("stop_var", "sum"),
            gross_exposure=("notional", "sum"),
            trades=("trade_id", "count"),
        )
        .sort_values("program_group")
    )
    for _, r in grouped.iterrows():
        program = str(r.get("program_group", ""))
        daily_pl = float(r.get("daily_pl", 0.0))
        unrealized_pl = float(r.get("unrealized_pl", 0.0))
        realized_pl = float(r.get("realized_pl", 0.0))
        var_to_stop = float(r.get("var_to_stop", 0.0))
        gross_exposure = float(r.get("gross_exposure", 0.0))
        trades = int(float(r.get("trades", 0)))

        st.markdown(f"**{program}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_metric_card("Total P/L", f"${daily_pl:,.2f}", _value_color(daily_pl), f"{trades} trade(s)"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Unrealized P/L", f"${unrealized_pl:,.2f}", _value_color(unrealized_pl)), unsafe_allow_html=True)
        c3.markdown(_metric_card("Realized P/L", f"${realized_pl:,.2f}", _value_color(realized_pl)), unsafe_allow_html=True)
        c4.markdown(_metric_card("VaR to Stop", f"${var_to_stop:,.2f}", "#F59E0B"), unsafe_allow_html=True)
        c5.markdown(_metric_card("Gross Exposure", f"${gross_exposure:,.2f}", "#60A5FA"), unsafe_allow_html=True)


def _render_portfolio_table(portfolio_df: pd.DataFrame) -> None:
    st.subheader("Portfolio Snapshot (Triggered Trades)")
    if portfolio_df.empty:
        st.info("No triggered trades available.")
        return

    work = portfolio_df.copy()
    if "program" not in work.columns:
        work["program"] = ""
    if "current_position" not in work.columns:
        work["current_position"] = pd.to_numeric(work.get("position", 0), errors="coerce").fillna(0.0)
    if "realized_pl" not in work.columns:
        work["realized_pl"] = 0.0
    if "unrealized_pl" not in work.columns:
        work["unrealized_pl"] = 0.0
    if "daily_pl" not in work.columns:
        work["daily_pl"] = pd.to_numeric(work.get("realized_pl", 0), errors="coerce").fillna(0.0) + pd.to_numeric(
            work.get("unrealized_pl", 0), errors="coerce"
        ).fillna(0.0)
    if "daily_pl_pct" not in work.columns:
        base = (
            pd.to_numeric(work.get("position", 0), errors="coerce").abs().fillna(0.0)
            * pd.to_numeric(work.get("entry_px", work.get("entry", 0)), errors="coerce").fillna(0.0)
            * pd.to_numeric(work.get("multiplier", 1.0), errors="coerce").fillna(1.0)
        ).replace(0, pd.NA)
        work["daily_pl_pct"] = (pd.to_numeric(work.get("daily_pl", 0), errors="coerce").fillna(0.0) / base) * 100.0
        work["unrealized_pl_pct"] = (pd.to_numeric(work.get("unrealized_pl", 0), errors="coerce").fillna(0.0) / base) * 100.0
        work["realized_pl_pct"] = (pd.to_numeric(work.get("realized_pl", 0), errors="coerce").fillna(0.0) / base) * 100.0
    if "unrealized_pl_pct" not in work.columns:
        work["unrealized_pl_pct"] = 0.0
    if "realized_pl_pct" not in work.columns:
        work["realized_pl_pct"] = 0.0
    if "state_label" not in work.columns:
        work["state_label"] = work.get("condition", "").astype(str)
    if "contract_code" not in work.columns:
        work["contract_code"] = ""
    if "notes" not in work.columns:
        work["notes"] = ""
    if "side" not in work.columns:
        work["side"] = ""
    if "source_file" not in work.columns:
        work["source_file"] = ""
    if "max_risk" not in work.columns:
        work["max_risk"] = 0.0
    if "max_reward" not in work.columns:
        work["max_reward"] = 0.0
    if "stop_var" not in work.columns:
        work["stop_var"] = 0.0
    if "multiplier" not in work.columns:
        work["multiplier"] = 1.0
    if "entry_px" not in work.columns:
        work["entry_px"] = pd.to_numeric(work.get("entry", 0), errors="coerce").fillna(0.0)
    if "market_px" not in work.columns:
        work["market_px"] = pd.to_numeric(work.get("market_price", work.get("quote_price", 0)), errors="coerce").fillna(0.0)
    if "session_open_px" not in work.columns:
        work["session_open_px"] = pd.to_numeric(work.get("session_open_price", 0), errors="coerce").fillna(0.0)
    if "cache_carry" not in work.columns:
        work["cache_carry"] = ""

    work = work.copy()
    work["_carry_rank"] = work["cache_carry"].astype(str).eq("1")
    work["_pos_rank"] = pd.to_numeric(work.get("current_position", 0), errors="coerce").abs().fillna(0.0)
    work = work.sort_values(by=["_carry_rank", "_pos_rank", "contract_code"], ascending=[False, False, True], kind="stable")
    work = work.drop(columns=["_carry_rank", "_pos_rank"])

    table_cols = [
        "program",
        "contract_code",
        "current_position",
        "daily_pl",
        "daily_pl_pct",
        "unrealized_pl",
        "unrealized_pl_pct",
        "realized_pl",
        "realized_pl_pct",
        "state_label",
        "side",
        "max_risk",
        "max_reward",
        "stop_var",
        "multiplier",
        "entry_px",
        "market_px",
        "session_open_px",
        "source_file",
        "notes",
    ]
    show = work[table_cols].copy()
    rename = {
        "source_file": "book",
        "contract_code": "contract",
        "program": "program",
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
        "unrealized_pl_pct": "unrealized_pl_%",
        "realized_pl_pct": "realized_pl_%",
        "daily_pl": "total_pl",
        "daily_pl_pct": "total_pl_%",
    }
    show = show.rename(columns=rename)
    state_color_map = work.get("state_color", pd.Series(["#E5E7EB"] * len(work), index=work.index)).to_dict()
    state_key_map = work.get("state_key", pd.Series(["UNKNOWN"] * len(work), index=work.index)).to_dict()
    dark_text_states = {
        "WAITING_ENTRY",
        "TRIGGERED",
        "T1_OPEN",
        "T2_OPEN",
        "HOLD_CLOSED",
        "NO_DATA",
        "DORMANT",
        "UNKNOWN",
    }

    def _portfolio_row_style(row: pd.Series):
        styles = [""] * len(row)
        # Match signal-viewer state coloring for the State column.
        if "state" in row.index:
            i = row.index.get_loc("state")
            bg = state_color_map.get(row.name, "#E5E7EB")
            state_key = str(state_key_map.get(row.name, "UNKNOWN"))
            fg = "#111827" if state_key in dark_text_states else "#FFFFFF"
            styles[i] = f"background-color: {bg}; color: {fg}; font-weight: 700;"

        # P/L color coding: green positive, red negative, gray zero.
        for col in ["total_pl", "total_pl_%", "unrealized_pl", "unrealized_pl_%", "realized_pl", "realized_pl_%"]:
            if col not in row.index:
                continue
            i = row.index.get_loc(col)
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                continue
            color = "#22C55E" if val > 0 else ("#EF4444" if val < 0 else "#9CA3AF")
            styles[i] = f"color: {color}; font-weight: 700;"
        return styles

    def _fmt_usd(v: object) -> str:
        num = pd.to_numeric(v, errors="coerce")
        if pd.isna(num):
            return ""
        return f"${float(num):,.2f}"

    def _fmt_pct(v: object) -> str:
        num = pd.to_numeric(v, errors="coerce")
        if pd.isna(num):
            return ""
        return f"{float(num):,.2f}%"

    formatters = {
        "total_pl": _fmt_usd,
        "unrealized_pl": _fmt_usd,
        "realized_pl": _fmt_usd,
        "max_risk": _fmt_usd,
        "max_reward": _fmt_usd,
        "var_to_stop": _fmt_usd,
        "entry": _fmt_usd,
        "current_price": _fmt_usd,
        "session_open": _fmt_usd,
        "total_pl_%": _fmt_pct,
        "unrealized_pl_%": _fmt_pct,
        "realized_pl_%": _fmt_pct,
        "current_position": "{:,.2f}",
        "multiplier": "{:,.2f}",
    }
    applicable_formatters = {k: v for k, v in formatters.items() if k in show.columns}

    st.dataframe(
        show.style.apply(_portfolio_row_style, axis=1).format(applicable_formatters).set_properties(**{"font-size": "14px"}),
        use_container_width=True,
        height=260,
        hide_index=True,
    )


def _render_var_to_stop_tab(portfolio_df: pd.DataFrame) -> None:
    st.subheader("VaR to Stop")

    if portfolio_df.empty:
        st.info("No triggered trades available.")
        return

    open_df = portfolio_df[
        (portfolio_df.get("condition", "") == "open_after_trigger")
        & (portfolio_df.get("current_position", 0).abs() > 0)
    ].copy()

    if open_df.empty:
        st.info("No open positions currently.")
        return

    total_var = float(open_df["stop_var"].sum())
    long_var = float(open_df.loc[open_df["current_position"] > 0, "stop_var"].sum())
    short_var = float(open_df.loc[open_df["current_position"] < 0, "stop_var"].sum())
    gross_now = float(open_df["notional"].sum())
    var_pct_gross = (total_var / gross_now * 100.0) if gross_now > 0 else 0.0
    top_var = float(open_df["stop_var"].max()) if len(open_df) else 0.0
    top_var_pct = (top_var / total_var * 100.0) if total_var > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(_metric_card("Total VaR to Stop", _abbr_number(total_var), "#F59E0B"), unsafe_allow_html=True)
    c2.markdown(_metric_card("Long VaR", _abbr_number(long_var), "#22C55E"), unsafe_allow_html=True)
    c3.markdown(_metric_card("Short VaR", _abbr_number(short_var), "#EF4444"), unsafe_allow_html=True)
    c4.markdown(_metric_card("VaR / Gross Exposure", f"{var_pct_gross:,.2f}%", "#60A5FA"), unsafe_allow_html=True)
    c5.markdown(
        _metric_card("Top Position VaR", _abbr_number(top_var), "#F59E0B", f"{top_var_pct:,.1f}% of total VaR"),
        unsafe_allow_html=True,
    )

    open_df["source_group"] = open_df["source_file"].astype(str).apply(_table_label)
    table_cols = [
        "source_file",
        "contract_code",
        "program",
        "side",
        "state_label",
        "current_position",
        "market_px",
        "effective_stop_px",
        "stop_var",
        "notional",
        "daily_pl",
        "notes",
    ]
    show_cols = [c for c in table_cols if c in open_df.columns]

    for book_name in ["Global Macro", "IntradayPlus"]:
        subset = open_df[open_df["source_group"] == book_name].copy()
        st.markdown(f"**{book_name}**")
        if subset.empty:
            st.caption("No open positions.")
            continue
        subset = subset.sort_values("stop_var", ascending=False)
        subset_show = subset[show_cols].rename(columns={"daily_pl": "total_pl"})
        st.dataframe(
            subset_show,
            use_container_width=True,
            hide_index=True,
            height=_table_height(len(subset), min_rows=8, max_rows=30),
        )


def _render_portfolio_admin_tab() -> None:
    st.subheader("Portfolio Admin")
    st.caption("Delete selected cached open positions without clearing all uploaded files.")
    cached_rows = _load_open_positions_cache()
    if not cached_rows:
        st.info("No cached open positions.")
        return

    cache_df = pd.DataFrame(cached_rows)
    if cache_df.empty:
        st.info("No cached open positions.")
        return

    cache_df = cache_df.copy()
    cache_df["cache_id"] = cache_df.apply(lambda r: _position_key(r.to_dict()), axis=1)
    show_cols = [
        "cache_id",
        "source_file",
        "contract_code",
        "program",
        "side",
        "position",
        "entry",
        "stop",
        "max_hold_date",
        "source_line",
    ]
    show_cols = [c for c in show_cols if c in cache_df.columns]
    selected = st.dataframe(
        cache_df[show_cols],
        use_container_width=True,
        hide_index=True,
        height=_table_height(len(cache_df), min_rows=8, max_rows=25),
        on_select="rerun",
        selection_mode="multi-row",
        key="tbl_cache_delete_select",
        column_config={"cache_id": None},
    )
    picked_rows: List[int] = []
    if selected and selected.selection and selected.selection.rows:
        picked_rows = [int(i) for i in selected.selection.rows]

    if st.button("Delete selected cached positions"):
        if not picked_rows:
            st.warning("Select one or more rows to delete.")
            return
        remove_ids = set(cache_df.iloc[picked_rows]["cache_id"].astype(str).tolist())
        remaining = [r for r in cached_rows if _position_key(r) not in remove_ids]
        _write_open_positions_cache_rows(remaining)
        st.success(f"Deleted {len(remove_ids)} cached position(s).")
        st.rerun()


def _style_table(df: pd.DataFrame):
    colors = df.get("state_color", pd.Series(["#FFFFFF"] * len(df))).tolist()
    state_keys = df.get("state_key", pd.Series(["UNKNOWN"] * len(df))).tolist()

    dark_text_states = {
        "WAITING_ENTRY",
        "TRIGGERED",
        "T1_OPEN",
        "T2_OPEN",
        "HOLD_CLOSED",
        "NO_DATA",
        "DORMANT",
        "UNKNOWN",
    }

    def _working_cols_for_row(row: pd.Series) -> List[str]:
        condition = str(row.get("condition", ""))
        try:
            targets_hit = int(float(row.get("targets_hit", 0)))
        except Exception:
            targets_hit = 0
        if condition != "open_after_trigger":
            return []
        if targets_hit <= 0:
            return ["stop", "target1"]
        if targets_hit == 1:
            return ["entry", "target2"]
        if targets_hit == 2:
            return ["target1", "target3"]
        return []

    def color_row(row):
        bg = colors[row.name]
        state_key = state_keys[row.name]
        fg = "#111827" if state_key in dark_text_states else "#FFFFFF"
        base_style = f"background-color: {bg}; color: {fg}; font-weight: 600;"
        styles = [base_style] * len(row)

        # Highlight currently working bracket orders (cell-level overlay).
        for col in _working_cols_for_row(row):
            if col not in row.index:
                continue
            i = row.index.get_loc(col)
            # Use explicit background fill; borders/shadows are unreliable in st.dataframe styles.
            styles[i] = "background-color: #FDE047; color: #111827; font-weight: 800;"
        return styles

    return df.style.apply(color_row, axis=1).set_properties(**{"font-size": "14px"})


def _table_height(row_count: int, min_rows: int = 15, max_rows: int = 60) -> int:
    rows = max(min_rows, min(max_rows, int(row_count)))
    return 42 + (rows * 35)


def _table_label(source_file: str) -> str:
    lower = source_file.lower()
    if "global_macro" in lower or "global macro" in lower:
        return "Global Macro"
    if lower in {"eq.csv", "eq"}:
        return "Global Macro"
    if "eqint" in lower:
        return "IntradayPlus"
    return source_file


def _sort_by_sheet_order(df: pd.DataFrame, file_order: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    order_map = {name: idx for idx, name in enumerate(file_order)}
    carry = work["cache_carry"] if "cache_carry" in work.columns else pd.Series([""] * len(work), index=work.index)
    work["_carry_rank"] = carry.astype(str).eq("1")
    work["_file_rank"] = work.get("source_file", "").astype(str).map(order_map).fillna(len(order_map)).astype(int)
    work["_line_rank"] = pd.to_numeric(work.get("source_line", ""), errors="coerce").fillna(10**9).astype(int)
    work = work.sort_values(
        by=["_carry_rank", "_file_rank", "_line_rank", "contract_code"],
        ascending=[False, True, True, True],
        kind="stable",
    ).drop(columns=["_carry_rank", "_file_rank", "_line_rank"])
    return work.reset_index(drop=True)


def _sort_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    triggered_conditions = {"open_after_trigger", "stopped_out", "all_targets_filled", "max_hold_closed"}
    work["_is_triggered"] = work.get("condition", "").isin(triggered_conditions)
    work["_state_rank"] = (
        work.get("state_key", "")
        .map(
            {
                "TRIGGERED": 0,
                "T1_OPEN": 1,
                "T2_OPEN": 2,
                "T1_STOP": 3,
                "T2_STOP": 4,
                "STOPPED": 5,
                "HOLD_CLOSED": 6,
                "T3_DONE": 7,
                "WAITING_ENTRY": 8,
                "NO_DATA": 9,
                "DORMANT": 10,
                "INVALID": 11,
                "UNKNOWN": 12,
            }
        )
        .fillna(99)
    )
    work = work.sort_values(
        by=["_is_triggered", "_state_rank", "source_file", "contract_code", "source_line"],
        ascending=[False, True, True, True, True],
    ).drop(columns=["_is_triggered", "_state_rank"])
    return work.reset_index(drop=True)


def _signal_table_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = [
        "program",
        "contract_code",
        "side",
        "status",
        "position",
        "entry",
        "stop",
        "target1",
        "target2",
        "target3",
        "state_label",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def _render_section(df_section: pd.DataFrame, section_title: str) -> None:
    st.subheader(section_title)
    df_show = _signal_table_order(df_section)
    counts = df_section["state_label"].value_counts().reset_index()
    counts.columns = ["State", "Count"]
    c1, c2 = st.columns([4, 1.3])
    with c1:
        st.dataframe(
            _style_table(df_show),
            use_container_width=True,
            height=_table_height(len(df_section)),
        )
    with c2:
        st.dataframe(counts, use_container_width=True, height=220)


def _render_single_selector(display_df: pd.DataFrame) -> None:
    st.subheader("Trade Selection")
    selector_df = _sort_for_selection(display_df)
    selector_df["source_group"] = selector_df["source_file"].astype(str).apply(_table_label)
    selector_cols = [
        "trade_id",
        "source_group",
        "contract_code",
        "program",
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
    today_open = datetime.combine(now_et.date(), time(hour=18, minute=0), tzinfo=ET)
    if now_et >= today_open:
        start_et = today_open
    else:
        start_et = today_open - timedelta(days=1)
    return start_et, now_et


def _choose_history_live_window(row: Dict[str, str], mode: str, history_cache: Dict[str, object]):
    start_et, end_et = _session_window_et()
    last_error = "No history candidates"
    best_symbol = ""
    best_hist = None
    best_score = None

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
            first_ts = pd.to_datetime(hist.index.min())
            start_gap_seconds = max(0.0, (first_ts - start_et).total_seconds())
            bar_count = len(hist)
            # Prefer earliest coverage from 6pm, then larger bar count.
            score = (start_gap_seconds, -bar_count)
            if best_score is None or score < best_score:
                best_score = score
                best_symbol = symbol
                best_hist = hist
                if start_gap_seconds <= 900 and bar_count >= 24:
                    # Good enough coverage (~15min from session open).
                    break
        last_error = f"No intraday bars for {symbol}"

    if best_hist is not None:
        return best_symbol, best_hist, ""
    return "", None, last_error


def _build_price_levels_chart(
    bars: pd.DataFrame,
    row: Dict[str, str],
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
):
    price_df = bars.reset_index().copy()
    x_col = price_df.columns[0]
    price_min = float(price_df["Low"].min())
    price_max = float(price_df["High"].max())
    x_scale = None
    if window_start is not None and window_end is not None:
        ws = pd.Timestamp(window_start)
        we = pd.Timestamp(window_end)
        if ws.tz is not None:
            ws = ws.tz_localize(None)
        if we.tz is not None:
            we = we.tz_localize(None)
        x_scale = alt.Scale(domain=[ws.to_pydatetime(), we.to_pydatetime()], nice=False)
    x_enc = alt.X(f"{x_col}:T", title="Time", scale=x_scale)

    level_specs = [
        ("entry", row.get("entry", ""), "#FDE047", "Entry"),
        ("stop", row.get("stop", ""), "#EF4444", "Stop"),
        ("target1", row.get("target1", ""), "#86EFAC", "Target 1"),
        ("target2", row.get("target2", ""), "#4ADE80", "Target 2"),
        ("target3", row.get("target3", ""), "#15803D", "Target 3"),
    ]

    level_rows = []
    for _, raw, color, label in level_specs:
        value = _parse_price_with_root(raw, row.get("root", ""))
        if value is not None:
            level_rows.append({"level": float(value), "label": label, "color": color})
            price_min = min(price_min, float(value))
            price_max = max(price_max, float(value))

    span = max(price_max - price_min, max(abs(price_max), 1.0) * 0.0025)
    pad = span * 0.12
    y_scale = alt.Scale(domain=[price_min - pad, price_max + pad], zero=False, nice=False)
    wick = (
        alt.Chart(price_df)
        .mark_rule(color="#94A3B8")
        .encode(
            x=x_enc,
            y=alt.Y("Low:Q", title="Price", scale=y_scale),
            y2=alt.Y2("High:Q"),
            tooltip=[
                alt.Tooltip(f"{x_col}:T"),
                alt.Tooltip("Open:Q"),
                alt.Tooltip("High:Q"),
                alt.Tooltip("Low:Q"),
                alt.Tooltip("Close:Q"),
            ],
        )
    )
    body = (
        alt.Chart(price_df)
        .mark_bar(size=7)
        .encode(
            x=x_enc,
            y=alt.Y("Open:Q", title="Price", scale=y_scale),
            y2=alt.Y2("Close:Q"),
            color=alt.condition("datum.Close >= datum.Open", alt.value("#22C55E"), alt.value("#EF4444")),
            tooltip=[
                alt.Tooltip(f"{x_col}:T"),
                alt.Tooltip("Open:Q"),
                alt.Tooltip("High:Q"),
                alt.Tooltip("Low:Q"),
                alt.Tooltip("Close:Q"),
            ],
        )
    )
    candles = wick + body

    marker_layer = None
    action_rows = _build_trade_action_rows(price_df, row, x_col)
    if action_rows:
        marker_df = pd.DataFrame(
            [{"ts": r["ts"], "px": r["px"], "dir": r["dir"], "event": r["event"]} for r in action_rows]
        )
        buy_df = marker_df[marker_df["dir"] == "BUY"]
        sell_df = marker_df[marker_df["dir"] == "SELL"]

        layers = []
        if not buy_df.empty:
            layers.append(
                alt.Chart(buy_df)
                .mark_point(shape="triangle-up", size=260, filled=True, color="#22C55E", stroke="#FFFFFF", strokeWidth=1.8)
                .encode(
                    x=alt.X("ts:T", scale=x_scale),
                    y=alt.Y("px:Q", scale=y_scale),
                    tooltip=[alt.Tooltip("event:N"), alt.Tooltip("dir:N"), alt.Tooltip("px:Q"), alt.Tooltip("ts:T")],
                )
            )
        if not sell_df.empty:
            layers.append(
                alt.Chart(sell_df)
                .mark_point(shape="triangle-down", size=260, filled=True, color="#DC2626", stroke="#FFFFFF", strokeWidth=1.8)
                .encode(
                    x=alt.X("ts:T", scale=x_scale),
                    y=alt.Y("px:Q", scale=y_scale),
                    tooltip=[alt.Tooltip("event:N"), alt.Tooltip("dir:N"), alt.Tooltip("px:Q"), alt.Tooltip("ts:T")],
                )
            )
        if layers:
            marker_layer = layers[0]
            for extra in layers[1:]:
                marker_layer = marker_layer + extra

    if not level_rows:
        chart = candles
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

    chart = candles + rules
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
    existing_uploaded_files = sorted(UPLOAD_DIR.glob("*.csv"))

    with st.sidebar:
        st.header("Controls")
        mode = st.selectbox("Yahoo mapping mode", ["both", "contract", "continuous"], index=0)
        intrabar_policy = st.selectbox("Intrabar priority", ["stop-first", "target-first"], index=0)
        live_only = st.checkbox("Live only (Send x)", value=True)
        include_quotes = st.checkbox("Include latest quote", value=False)
        auto_refresh = st.checkbox("Enable auto refresh", value=False)
        refresh_minutes = st.slider("Refresh minutes", min_value=1, max_value=60, value=5, step=1)
        manual_refresh = st.button("Refresh now")
        clear_files = st.button("Clear uploaded files")
        open_cache_file = st.file_uploader("Load Open Position Cache CSV", type=["csv"], accept_multiple_files=False)
        st.caption("Delete individual uploaded file")
        delete_target = st.selectbox(
            "Select file to delete",
            options=[""] + [p.name for p in existing_uploaded_files],
            format_func=lambda x: x if x else "-- choose file --",
            key="delete_target_file",
        )
        delete_selected_file = st.button("Delete selected file", disabled=(delete_target == ""))

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
        OPEN_CACHE_PATH.unlink(missing_ok=True)
        st.success("Cleared saved uploaded files.")
        st.rerun()

    if open_cache_file is not None:
        OPEN_CACHE_PATH.write_bytes(open_cache_file.getvalue())
        st.success("Loaded open position cache file.")
        st.rerun()

    if delete_selected_file and delete_target:
        target_path = UPLOAD_DIR / delete_target
        target_path.unlink(missing_ok=True)
        st.success(f"Deleted uploaded file: {delete_target}")
        st.rerun()

    if uploaded_files:
        saved_paths = _save_uploaded_files(uploaded_files)
    else:
        saved_paths = sorted(UPLOAD_DIR.glob("*.csv"))

    if not saved_paths:
        st.info("Upload your trade CSV files to start the live table.")
        return

    st.caption("Using uploaded files: " + ", ".join(p.name for p in saved_paths))
    input_rows = _load_rows_from_saved_files(saved_paths)
    cached_open_rows = _load_open_positions_cache()
    if cached_open_rows:
        input_rows = _merge_with_open_cache(input_rows, cached_open_rows)
        st.caption(f"Including {len(cached_open_rows)} open cached position(s) from prior sessions.")

    if auto_refresh and HAS_ST_AUTOREFRESH:
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="trade_table_refresh")
    elif auto_refresh and not HAS_ST_AUTOREFRESH:
        st.info("Install `streamlit-autorefresh` for smooth in-app refresh (no page blink): `pip install streamlit-autorefresh`")

    df = _analyze(
        input_rows=input_rows,
        mode=mode,
        intrabar_policy=intrabar_policy,
        live_only=live_only,
        include_quotes=include_quotes,
    )
    df = _sort_by_sheet_order(df, [p.name for p in saved_paths])

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
    display_df = _ensure_state_columns(display_df)
    st.session_state.last_df = display_df
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
    _save_open_positions_cache(portfolio_df)
    tab_live, tab_var, tab_risk, tab_admin = st.tabs(["Live Monitor", "VaR to Stop", "Pre-Trade Risk", "Portfolio Admin"])

    with tab_live:
        st.subheader("Portfolio Stats")
        _render_stat_cards(portfolio_df, exposure_stats)
        _render_program_stats(portfolio_df)
        _render_portfolio_table(portfolio_df)

        if "source_file" not in display_df.columns:
            _render_section(display_df.reset_index(drop=True), "All Trades")
        else:
            for source_file, section in display_df.groupby("source_file", sort=False):
                _render_section(section.reset_index(drop=True), _table_label(str(source_file)))

        st.subheader("Color Legend")
        legend = pd.DataFrame(
            [
                {"State": "Live, waiting entry", "Color": "Yellow"},
                {"State": "Entry Triggered, working T1 + stop", "Color": "Orange"},
                {"State": "Target1 Filled, working T2 + stop@entry", "Color": "Light Blue"},
                {"State": "Target2 Filled, working T3 + stop@entry", "Color": "Blue"},
                {"State": "Target 1 Filled, Stop @ Entry", "Color": "Green"},
                {"State": "Target 2 Filled, Stop @ Target 1", "Color": "Dark Green"},
                {"State": "All targets Filled", "Color": "Darkest Green"},
                {"State": "Hard Stop", "Color": "Red"},
                {"State": "Max hold close (profit/flat/loss)", "Color": "Green/Gray/Red"},
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
                expected_start, _ = _session_window_et()
                loaded_start = pd.to_datetime(bars.index.min()).tz_convert(ET)
                loaded_end = pd.to_datetime(bars.index.max()).tz_convert(ET)
                st.caption(
                    "Bars loaded: "
                    f"{loaded_start.strftime('%Y-%m-%d %I:%M %p ET')} to "
                    f"{loaded_end.strftime('%Y-%m-%d %I:%M %p ET')}"
                )
                if loaded_start > expected_start + timedelta(minutes=15):
                    st.warning(
                        "Data source started after the 6:00 PM ET session open for this symbol. "
                        "Using best available intraday coverage."
                    )
            except Exception:
                pass
            st.altair_chart(
                _build_price_levels_chart(bars, selected_row, start_et, end_et),
                use_container_width=True,
            )

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

    with tab_var:
        _render_var_to_stop_tab(portfolio_df)

    with tab_admin:
        _render_portfolio_admin_tab()


if __name__ == "__main__":
    main()
