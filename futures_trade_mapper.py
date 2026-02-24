#!/usr/bin/env python3
"""Analyze futures trade sheets against historical price data.

Workflow:
- Read daily trade CSVs in this folder (skip metadata line, then header line).
- Parse live/dormant status from STATUS (e.g. "Send 1.0").
- Restrict to US futures roots (Yahoo-compatible mapping).
- Pull historical bars with yfinance.
- Simulate order lifecycle:
  - Entry via stop-market at ENTRY level.
  - If triggered: submit stop + targets.
  - Target sizing: T1=50%, T2=25%, T3=25%.
  - After T1 fills, stop quantity is reduced and stop price moves to ENTRY.
  - After T2 fills, stop quantity reduced again (stop price remains ENTRY).
- Produce CSV with condition + notes to show what is currently being worked.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: yfinance. Install with `pip install yfinance`.") from exc


logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)

MONTH_CODES = set("FGHJKMNQUVXZ")
CONTRACT_RE = re.compile(r"^\s*([A-Z0-9]+)\s+([FGHJKMNQUVXZ])(\d{1,2})\b")
SEND_RE = re.compile(r"send\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
SIDE_RE = re.compile(r"\(([^(]*)\)")

US_CONTINUOUS_SYMBOLS: Dict[str, str] = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "MNQ": "MNQ=F",
    "YM": "YM=F",
    "NKD": "NKD=F",
    "RTY": "RTY=F",
    "CL": "CL=F",
    "NG": "NG=F",
    "RB": "RB=F",
    "HO": "HO=F",
    "GC": "GC=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "PL": "PL=F",
    "PA": "PA=F",
    "ZB": "ZB=F",
    "ZN": "ZN=F",
    "ZF": "ZF=F",
    "ZT": "ZT=F",
    "KE": "KE=F",
    "ZW": "ZW=F",
    "ZC": "ZC=F",
    "ZS": "ZS=F",
    "LE": "LE=F",
    "HE": "HE=F",
    "6E": "6E=F",
    "6B": "6B=F",
    "6J": "6J=F",
    "6C": "6C=F",
    "6A": "6A=F",
    "6S": "6S=F",
}

US_EXCHANGE_SUFFIXES: Dict[str, List[str]] = {
    "ES": [".CME"],
    "NQ": [".CME"],
    "MNQ": [".CME"],
    "YM": [".CBT", ".CME"],
    "NKD": [".CME"],
    "RTY": [".CME"],
    "CL": [".NYM"],
    "NG": [".NYM"],
    "RB": [".NYM"],
    "HO": [".NYM"],
    "GC": [".CMX", ".COMEX"],
    "SI": [".CMX", ".COMEX"],
    "HG": [".CMX", ".COMEX"],
    "PL": [".NYM"],
    "PA": [".NYM"],
    "ZB": [".CBT"],
    "ZN": [".CBT"],
    "ZF": [".CBT"],
    "ZT": [".CBT"],
    "KE": [".CBT"],
    "ZW": [".CBT"],
    "ZC": [".CBT"],
    "ZS": [".CBT"],
    "LE": [".CME"],
    "HE": [".CME"],
    "6E": [".CME"],
    "6B": [".CME"],
    "6J": [".CME"],
    "6C": [".CME"],
    "6A": [".CME"],
    "6S": [".CME"],
}

CONTINUOUS_SYMBOL_CANDIDATES: Dict[str, List[str]] = {
    # If MNQ is not available from source, fall back to NQ for market data.
    "MNQ": ["MNQ=F", "NQ=F"],
    # Nikkei aliases vary by source availability.
    "NKD": ["NKD=F", "NIY=F", "N225=F"],
}

MONTH_CODE_TO_NAME: Dict[str, str] = {
    "F": "Jan",
    "G": "Feb",
    "H": "Mar",
    "J": "Apr",
    "K": "May",
    "M": "Jun",
    "N": "Jul",
    "Q": "Aug",
    "U": "Sep",
    "V": "Oct",
    "X": "Nov",
    "Z": "Dec",
}


@dataclass
class ParsedContract:
    root: str
    month_code: str
    year: int


@dataclass
class QuoteResult:
    symbol: str
    price: Optional[float]
    timestamp: Optional[str]
    source: str
    error: Optional[str]


@dataclass
class TradeState:
    condition: str
    notes: str
    working_stop_price: Optional[float]
    working_stop_size: float
    next_target: Optional[str]
    next_target_price: Optional[float]
    remaining_size: float
    targets_hit: int


def _clean_row(raw: Dict[str, str]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for key, value in raw.items():
        if key is None:
            continue
        clean[key.strip()] = (value or "").strip()
    return clean


def parse_float(text: str) -> Optional[float]:
    v = text.strip()
    if not v or v == "-":
        return None
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def parse_price(text: str, root: Optional[str] = None) -> Optional[float]:
    """Parse price text with root-aware handling.

    For ZN/ZB levels in sheet format (e.g. 113.240), treat the 3-digit
    fractional block as 32nds-tenths:
    - 113.240 -> 113 + 24.0/32
    - 113.025 -> 113 + 2.5/32
    """
    v = text.strip()
    if not v or v == "-":
        return None

    # 32nds encoding support for CBOT rates where levels are in 1/32 increments.
    if (root or "").upper() in {"ZN", "ZB"}:
        # Accept both:
        # - 3-digit tenths-of-32nds (e.g. 113.145 -> 14.5/32)
        # - 2-digit whole 32nds (e.g. 118.11 -> 11/32)
        match = re.fullmatch(r"\s*([+-]?\d+)\.(\d{2,3})\s*", v)
        if match:
            whole_text, frac_text = match.groups()
            try:
                whole = int(whole_text)
                frac_32 = int(frac_text) / 10.0 if len(frac_text) == 3 else float(int(frac_text))
                sign = -1.0 if whole < 0 else 1.0
                return float(whole) + sign * (frac_32 / 32.0)
            except ValueError:
                pass

    return parse_float(v)


def infer_year(yy_text: str, reference_year: Optional[int]) -> int:
    if len(yy_text) == 2:
        return 2000 + int(yy_text)
    digit = int(yy_text)
    base_decade = 2020 if reference_year is None else (reference_year // 10) * 10
    candidates = [base_decade - 10 + digit, base_decade + digit, base_decade + 10 + digit]
    if reference_year is None:
        return candidates[1]
    return sorted(candidates, key=lambda y: (abs(y - reference_year), y < reference_year))[0]


def parse_contract(contract_text: str, reference_year: Optional[int]) -> Optional[ParsedContract]:
    match = CONTRACT_RE.search(contract_text)
    if not match:
        return None
    root, month_code, yy = match.groups()
    if month_code not in MONTH_CODES:
        return None
    return ParsedContract(root=root, month_code=month_code, year=infer_year(yy, reference_year))


def parse_status(status_text: str) -> Tuple[bool, Optional[float]]:
    match = SEND_RE.search(status_text)
    if not match:
        return False, None
    return True, float(match.group(1))


def format_contract_code(parsed: Optional[ParsedContract]) -> str:
    if parsed is None:
        return ""
    yy = str(parsed.year)[-2:]
    return f"{parsed.root}{parsed.month_code}{yy}"


def format_contract_name(parsed: Optional[ParsedContract]) -> str:
    if parsed is None:
        return ""
    month_name = MONTH_CODE_TO_NAME.get(parsed.month_code, parsed.month_code)
    return f"{parsed.root} {month_name} {parsed.year}"


def format_futures_clean(parsed: Optional[ParsedContract]) -> str:
    if parsed is None:
        return ""
    yy_short = str(parsed.year % 10)
    return f"{parsed.root} {parsed.month_code}{yy_short}"


def parse_position(position_text: str) -> Optional[float]:
    text = position_text.strip()
    if not text or text == "-":
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.strip("()").replace(",", "").strip()
    if not cleaned or cleaned == "-":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def parse_side(contract_text: str) -> Optional[str]:
    block = SIDE_RE.search(contract_text)
    if not block:
        return None
    inner = block.group(1)
    # Supports L1/S1 and LM/SM tags.
    match = re.search(r"\b([LS])(?:\d+|M)\b", inner)
    return match.group(1) if match else None


def parse_program(contract_text: str) -> str:
    block = SIDE_RE.search(contract_text)
    if not block:
        return ""
    # Normalize "INT+ L1" -> "INT+L1", keep variants like "L1", "SM".
    return re.sub(r"\s+", "", block.group(1).strip())


def parse_file_date(metadata: str) -> Optional[date]:
    parts = [p.strip() for p in metadata.split(",")]
    if len(parts) < 3:
        return None
    if not (parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()):
        return None
    month = int(parts[0])
    day = int(parts[1])
    year = int(parts[2])
    try:
        return date(year, month, day)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        metadata = next(handle, "").strip()
        file_date = parse_file_date(metadata)
        reference_year = file_date.year if file_date else None
        reader = csv.DictReader(handle)

        for idx, raw in enumerate(reader, start=3):
            row = _clean_row(raw)
            contract_text = row.get("Future (System Direction)", "") or row.get("FUTURE", "")
            if not contract_text:
                continue

            parsed = parse_contract(contract_text, reference_year)
            # Ignore footer/open-position/non-signal rows.
            if parsed is None:
                continue
            if parse_float(row.get("ENTRY", "")) is None:
                continue
            if parse_float(row.get("STOP", "")) is None:
                continue
            if parse_float(row.get("Target1", "")) is None:
                continue
            if parse_float(row.get("Target2", "")) is None:
                continue
            if parse_float(row.get("Target3", "")) is None:
                continue
            is_live, send_mult = parse_status(row.get("STATUS", ""))
            signal_date = file_date.isoformat() if file_date else ""

            rows.append(
                {
                    "source_file": csv_path.name,
                    "source_line": str(idx),
                    "signal_date": signal_date,
                    "contract_text": contract_text,
                    "program": parse_program(contract_text),
                    "futures_clean": format_futures_clean(parsed),
                    "contract_code": format_contract_code(parsed),
                    "contract_name": format_contract_name(parsed),
                    "root": parsed.root if parsed else "",
                    "month_code": parsed.month_code if parsed else "",
                    "year": str(parsed.year) if parsed else "",
                    "side": parse_side(contract_text) or "",
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
                    "position": ""
                    if parse_position(row.get("Position", "")) is None
                    else str(parse_position(row.get("Position", ""))),
                    "account": row.get("Account", ""),
                    "max_hold_date": row.get("Max Hold Date", ""),
                }
            )
    return rows


def contract_candidates(parsed: ParsedContract) -> List[str]:
    yy = str(parsed.year)[-2:]
    base = f"{parsed.root}{parsed.month_code}{yy}"
    suffixes = US_EXCHANGE_SUFFIXES.get(parsed.root, [])
    return [base] + [f"{base}{suffix}" for suffix in suffixes]


def fetch_quote(symbol: str) -> QuoteResult:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty and "Close" in hist.columns:
            close = hist["Close"].dropna()
            if not close.empty:
                return QuoteResult(symbol=symbol, price=float(close.iloc[-1]), timestamp=str(close.index[-1]), source="history.close", error=None)

        fast = getattr(ticker, "fast_info", None)
        if fast:
            for key in ("lastPrice", "regularMarketPrice", "previousClose"):
                value = fast.get(key)
                if value is not None:
                    return QuoteResult(symbol=symbol, price=float(value), timestamp=None, source=f"fast_info.{key}", error=None)

        return QuoteResult(symbol=symbol, price=None, timestamp=None, source="none", error="No price found")
    except Exception as exc:  # pragma: no cover
        return QuoteResult(symbol=symbol, price=None, timestamp=None, source="error", error=str(exc))


def resolve_quote(row: Dict[str, str], mode: str, quote_cache: Dict[str, QuoteResult]) -> QuoteResult:
    root = row.get("root", "")
    month_code = row.get("month_code", "")
    year_text = row.get("year", "")

    candidates: List[str] = []
    if mode in {"contract", "both"} and root and month_code and year_text:
        parsed = ParsedContract(root=root, month_code=month_code, year=int(year_text))
        candidates.extend(contract_candidates(parsed))
    if mode in {"continuous", "both"} and root in US_CONTINUOUS_SYMBOLS:
        candidates.extend(CONTINUOUS_SYMBOL_CANDIDATES.get(root, [US_CONTINUOUS_SYMBOLS[root]]))

    if not candidates:
        return QuoteResult(symbol="", price=None, timestamp=None, source="none", error="No Yahoo candidate symbols")

    seen = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    last_error: Optional[str] = None
    for symbol in ordered:
        if symbol not in quote_cache:
            quote_cache[symbol] = fetch_quote(symbol)
        quote = quote_cache[symbol]
        if quote.price is not None:
            return quote
        if quote.error:
            last_error = quote.error

    return QuoteResult(symbol=ordered[-1], price=None, timestamp=None, source="none", error=last_error or "No valid price")


def fetch_bars(symbol: str, start: date, end: date):
    ticker = yf.Ticker(symbol)
    end_exclusive = end + timedelta(days=1)
    hist = ticker.history(start=start.isoformat(), end=end_exclusive.isoformat(), interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return None
    if not {"Open", "High", "Low", "Close"}.issubset(set(hist.columns)):
        return None
    return hist


def resolve_history_symbol(row: Dict[str, str], mode: str) -> List[str]:
    root = row.get("root", "")
    month_code = row.get("month_code", "")
    year_text = row.get("year", "")

    candidates: List[str] = []
    if mode in {"contract", "both"} and root and month_code and year_text:
        parsed = ParsedContract(root=root, month_code=month_code, year=int(year_text))
        candidates.extend(contract_candidates(parsed))
    if mode in {"continuous", "both"} and root in US_CONTINUOUS_SYMBOLS:
        candidates.extend(CONTINUOUS_SYMBOL_CANDIDATES.get(root, [US_CONTINUOUS_SYMBOLS[root]]))

    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def bar_touches_entry(side: str, entry: float, high: float, low: float) -> bool:
    if side == "L":
        return high >= entry
    if side == "S":
        return low <= entry
    return False


def bar_hits_stop(side: str, stop: float, high: float, low: float) -> bool:
    if side == "L":
        return low <= stop
    if side == "S":
        return high >= stop
    return False


def bar_hits_target(side: str, target: float, high: float, low: float) -> bool:
    if side == "L":
        return high >= target
    if side == "S":
        return low <= target
    return False


def simulate_trade(row: Dict[str, str], bars, intrabar_policy: str) -> TradeState:
    side_raw = str(row.get("side", "")).upper()
    side = {"BUY": "L", "SELL": "S"}.get(side_raw, side_raw)
    root = row.get("root", "")
    entry = parse_price(row.get("entry", ""), root)
    stop = parse_price(row.get("stop", ""), root)
    t1 = parse_price(row.get("target1", ""), root)
    t2 = parse_price(row.get("target2", ""), root)
    t3 = parse_price(row.get("target3", ""), root)

    if side not in {"L", "S"} or None in {entry, stop, t1, t2, t3}:
        return TradeState(
            condition="invalid_levels",
            notes="Missing side or numeric levels.",
            working_stop_price=None,
            working_stop_size=0.0,
            next_target=None,
            next_target_price=None,
            remaining_size=0.0,
            targets_hit=0,
        )

    pending = [("target1", float(t1), 0.50), ("target2", float(t2), 0.25), ("target3", float(t3), 0.25)]

    entered = False
    remaining = 1.0
    stop_price = float(stop)
    notes: List[str] = []

    for idx in range(len(bars.index)):
        ts = bars.index[idx]
        day = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        high = float(bars.iloc[idx]["High"])
        low = float(bars.iloc[idx]["Low"])

        if not entered:
            if not bar_touches_entry(side, float(entry), high, low):
                continue
            entered = True
            notes.append(f"triggered on {day}")

        # Once entered, process events for this bar in chosen priority.
        while remaining > 0:
            stop_hit = bar_hits_stop(side, stop_price, high, low)
            next_hit = False
            if pending:
                next_hit = bar_hits_target(side, pending[0][1], high, low)

            if not stop_hit and not next_hit:
                break

            if stop_hit and next_hit:
                notes.append(f"intrabar ambiguity on {day}; used {intrabar_policy}")

            if (intrabar_policy == "stop-first" and stop_hit) or (intrabar_policy == "target-first" and not next_hit):
                notes.append(f"stopped out on {day} at {stop_price:.5g}")
                return TradeState(
                    condition="stopped_out",
                    notes=", ".join(notes),
                    working_stop_price=None,
                    working_stop_size=0.0,
                    next_target=None,
                    next_target_price=None,
                    remaining_size=0.0,
                    targets_hit=3 - len(pending),
                )

            if next_hit and pending:
                target_name, target_px, target_size = pending.pop(0)
                remaining = max(0.0, remaining - target_size)
                notes.append(f"{target_name} hit on {day} at {target_px:.5g}")
                if target_name == "target1":
                    stop_price = float(entry)
                    notes.append("stop moved to entry after target1")
                if remaining <= 0:
                    return TradeState(
                        condition="all_targets_filled",
                        notes=", ".join(notes),
                        working_stop_price=None,
                        working_stop_size=0.0,
                        next_target=None,
                        next_target_price=None,
                        remaining_size=0.0,
                        targets_hit=3,
                    )

            # For target-first, stop could still be hit after filling current target on same bar.
            if intrabar_policy == "target-first":
                post_stop_hit = bar_hits_stop(side, stop_price, high, low)
                if post_stop_hit:
                    notes.append(f"stopped out on {day} at {stop_price:.5g}")
                    return TradeState(
                        condition="stopped_out",
                        notes=", ".join(notes),
                        working_stop_price=None,
                        working_stop_size=0.0,
                        next_target=None,
                        next_target_price=None,
                        remaining_size=0.0,
                        targets_hit=3 - len(pending),
                    )

            if intrabar_policy == "stop-first" and stop_hit and not next_hit:
                notes.append(f"stopped out on {day} at {stop_price:.5g}")
                return TradeState(
                    condition="stopped_out",
                    notes=", ".join(notes),
                    working_stop_price=None,
                    working_stop_size=0.0,
                    next_target=None,
                    next_target_price=None,
                    remaining_size=0.0,
                    targets_hit=3 - len(pending),
                )

            # If next target not hit after current update and stop not hit, bar processing is complete.
            if not pending or not bar_hits_target(side, pending[0][1], high, low):
                break

    if not entered:
        return TradeState(
            condition="waiting_entry",
            notes="entry not triggered in analysis window",
            working_stop_price=None,
            working_stop_size=0.0,
            next_target="target1",
            next_target_price=t1,
            remaining_size=1.0,
            targets_hit=0,
        )

    next_target = pending[0][0] if pending else None
    next_target_price = pending[0][1] if pending else None
    return TradeState(
        condition="open_after_trigger",
        notes=", ".join(notes) if notes else "triggered; still open",
        working_stop_price=stop_price,
        working_stop_size=remaining,
        next_target=next_target,
        next_target_price=next_target_price,
        remaining_size=remaining,
        targets_hit=3 - len(pending),
    )


def filter_us_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if row.get("root") in US_CONTINUOUS_SYMBOLS]


def parse_iso_date(text: str) -> Optional[date]:
    t = text.strip()
    if not t:
        return None
    try:
        return datetime.strptime(t, "%Y-%m-%d").date()
    except ValueError:
        return None


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, str]]) -> None:
    cols = [
        ("contract_name", 18),
        ("contract_code", 10),
        ("status", 10),
        ("condition", 19),
        ("working_stop_price", 12),
        ("next_target", 10),
        ("next_target_price", 12),
        ("notes", 70),
    ]
    header = " | ".join(name.ljust(width) for name, width in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for name, width in cols:
            val = str(row.get(name, ""))
            if len(val) > width:
                val = val[: width - 3] + "..."
            values.append(val.ljust(width))
        print(" | ".join(values))


def choose_history(row: Dict[str, str], mode: str, history_cache: Dict[str, object]):
    signal_date = parse_iso_date(row.get("signal_date", ""))
    max_hold = parse_iso_date(row.get("max_hold_date", ""))
    if signal_date is None or max_hold is None:
        return "", None, "Missing signal_date or max_hold_date"
    if max_hold < signal_date:
        max_hold = signal_date

    last_error = "No history candidates"
    for symbol in resolve_history_symbol(row, mode):
        cache_key = f"{symbol}|{signal_date.isoformat()}|{max_hold.isoformat()}"
        if cache_key not in history_cache:
            try:
                hist = yf.Ticker(symbol).history(
                    start=signal_date.isoformat(),
                    end=(max_hold + timedelta(days=1)).isoformat(),
                    interval="1d",
                    auto_adjust=False,
                )
                history_cache[cache_key] = hist if hist is not None and not hist.empty else None
            except Exception as exc:  # pragma: no cover
                history_cache[cache_key] = None
                last_error = str(exc)
        hist = history_cache[cache_key]
        if hist is not None and {"High", "Low", "Open", "Close"}.issubset(set(hist.columns)):
            return symbol, hist, ""
        last_error = f"No bars for {symbol}"

    return "", None, last_error


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default="*.csv", help="Input trade CSV glob (default: *.csv)")
    parser.add_argument("--output", default="futures_trade_states.csv", help="Output CSV (default: futures_trade_states.csv)")
    parser.add_argument("--live-only", action="store_true", help="Only rows with STATUS like Send x")
    parser.add_argument("--mode", choices=["contract", "continuous", "both"], default="both", help="Yahoo symbol resolution mode")
    parser.add_argument(
        "--intrabar-policy",
        choices=["stop-first", "target-first"],
        default="stop-first",
        help="When stop and target touch in same bar, choose resolution policy",
    )
    parser.add_argument("--skip-quotes", action="store_true", help="Skip latest quote lookup")
    parser.add_argument("--skip-history", action="store_true", help="Skip historical lifecycle analysis")
    parser.add_argument("--print-table", dest="print_table", action="store_true", default=True, help="Print compact table to stdout (default: on)")
    parser.add_argument("--no-print-table", dest="print_table", action="store_false", help="Disable table printing")
    args = parser.parse_args()

    input_pattern = Path(args.input_glob)
    if input_pattern.is_absolute():
        glob_pattern = args.input_glob
    else:
        glob_pattern = str(script_dir / args.input_glob)
    csv_paths = [Path(p) for p in sorted(glob.glob(glob_pattern))]
    if not csv_paths:
        raise SystemExit(f"No files matched pattern: {args.input_glob}")

    all_rows: List[Dict[str, str]] = []
    for path in csv_paths:
        all_rows.extend(load_rows(path))

    rows = filter_us_rows(all_rows)
    if args.live_only:
        rows = [r for r in rows if r.get("is_live") == "True"]

    quote_cache: Dict[str, QuoteResult] = {}
    history_cache: Dict[str, object] = {}
    output_rows: List[Dict[str, str]] = []

    for row in rows:
        out = dict(row)

        if args.skip_quotes:
            quote = QuoteResult(symbol=US_CONTINUOUS_SYMBOLS.get(row.get("root", ""), ""), price=None, timestamp=None, source="skipped", error=None)
        else:
            quote = resolve_quote(row, args.mode, quote_cache)

        out["yahoo_symbol"] = quote.symbol
        out["quote_price"] = "" if quote.price is None else str(quote.price)
        out["quote_timestamp"] = quote.timestamp or ""
        out["quote_source"] = quote.source
        out["quote_error"] = quote.error or ""

        if args.skip_history:
            state = TradeState(
                condition="history_skipped",
                notes="Historical lifecycle analysis skipped",
                working_stop_price=None,
                working_stop_size=0.0,
                next_target=None,
                next_target_price=None,
                remaining_size=0.0,
                targets_hit=0,
            )
            history_symbol = ""
            history_error = ""
        else:
            history_symbol, bars, history_error = choose_history(row, args.mode, history_cache)
            if bars is None:
                state = TradeState(
                    condition="history_unavailable",
                    notes=history_error,
                    working_stop_price=None,
                    working_stop_size=0.0,
                    next_target=None,
                    next_target_price=None,
                    remaining_size=0.0,
                    targets_hit=0,
                )
            else:
                state = simulate_trade(row, bars, args.intrabar_policy)

        out["history_symbol"] = history_symbol
        out["condition"] = state.condition
        out["working_stop_price"] = "" if state.working_stop_price is None else str(state.working_stop_price)
        out["working_stop_size"] = str(state.working_stop_size)
        out["next_target"] = state.next_target or ""
        out["next_target_price"] = "" if state.next_target_price is None else str(state.next_target_price)
        out["remaining_size"] = str(state.remaining_size)
        out["targets_hit"] = str(state.targets_hit)
        out["notes"] = state.notes

        highlight = ""
        if state.condition == "waiting_entry":
            highlight = "YELLOW"
        elif state.condition == "open_after_trigger":
            highlight = "GREEN"
        elif state.condition == "stopped_out":
            highlight = "RED"
        elif state.condition == "all_targets_filled":
            highlight = "BLUE"
        out["highlight"] = highlight

        output_rows.append(out)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    write_csv(output_rows, output_path)

    if args.print_table:
        print_table(output_rows)

    print(f"Loaded rows: {len(all_rows)}")
    print(f"US rows: {len(rows)}")
    print(f"Wrote: {output_path.resolve()}")


if __name__ == "__main__":
    main()
