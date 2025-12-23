#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------------------
# Normalization / matching
# ----------------------------

STOPWORDS = {
    "manual", "operation", "operating", "owners", "owner's", "owner",
    "instruction", "instructions", "user", "users", "user's", "guide",
    "reference", "quick", "start", "basic", "service", "maintenance",
    "schematics", "with", "and", "&", "for", "the", "a", "an", "series",
    "digital", "camera", "transceiver", "sewing", "machine",
    "mark", "mk", "mkii", "ii", "iii", "iv", "v", "vi", "vii", "viii",
    "cd", "dvd", "blu", "ray", "black",
}

ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
}

RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
RE_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")

    tokens = re.split(r"\s+", s)
    tokens2 = []
    for t in tokens:
        t_clean = RE_NON_ALNUM.sub("", t)
        if t_clean in ROMAN_MAP:
            tokens2.append(ROMAN_MAP[t_clean])
        else:
            tokens2.append(t)
    s = " ".join(tokens2)

    s = RE_NON_ALNUM.sub(" ", s)
    s = RE_MULTI_SPACE.sub(" ", s).strip()
    return s


def token_set(s: str) -> List[str]:
    s = normalize_text(s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small & big) / len(small) if small else 0.0


def score_match(title: str, key: str) -> float:
    t_tokens = token_set(title)
    k_tokens = token_set(key)

    j = jaccard(t_tokens, k_tokens)
    c = containment(t_tokens, k_tokens)

    nt = normalize_text(title)
    nk = normalize_text(key)

    substr_bonus = 0.0
    if nk and nk in nt:
        substr_bonus = 0.10
    elif nt and nt in nk:
        substr_bonus = 0.07

    raw = 0.55 * j + 0.35 * c + substr_bonus
    raw = max(0.0, min(1.0, raw))
    return 100.0 * raw


def top_n_matches(title: str, candidates: List[str], n: int = 3) -> List[Tuple[str, float]]:
    scored = [(k, score_match(title, k)) for k in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


# ----------------------------
# I/O structures
# ----------------------------

@dataclass
class OrderRow:
    order_number: str
    item_id: str
    title: str
    item_url: str


def read_orders_csv(path: Path) -> List[OrderRow]:
    rows: List[OrderRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(OrderRow(
                order_number=(d.get("order_number") or "").strip(),
                item_id=(d.get("item_id") or "").strip(),
                title=(d.get("title") or "").strip(),
                item_url=(d.get("item_url") or "").strip(),
            ))
    return rows


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_pdf_basenames(pdf_dir: Path, recursive: bool = False) -> List[str]:
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(pdf_dir.glob(pattern))
    basenames = [p.stem for p in pdfs if p.is_file()]
    return basenames


# ----------------------------
# Matching / update logic
# ----------------------------

@dataclass
class MatchResult:
    order_number: str
    item_id: str
    best_score: float
    second_score: float
    margin: float
    best_key: str
    second_key: str
    third_key: str
    third_score: float
    action: str
    title: str
    item_url: str
    previous_url: str


def should_auto_update(best: float, second: float, min_score: float, min_margin: float) -> bool:
    return (best >= min_score) and ((best - second) >= min_margin)


def update_links_with_margin(
    orders: List[OrderRow],
    pdf_keys: List[str],
    link_db: Dict[str, dict],
    min_score: float,
    min_margin: float,
    overwrite: bool,
) -> Tuple[List[MatchResult], Dict[str, dict]]:
    updated = dict(link_db)
    results: List[MatchResult] = []

    for o in orders:
        top3 = top_n_matches(o.title, pdf_keys, n=3)

        # pad if fewer
        while len(top3) < 3:
            top3.append(("", -1.0))

        (k1, s1), (k2, s2), (k3, s3) = top3
        margin = (s1 - s2) if (s2 >= 0) else s1

        prev_url = ""
        if k1 and k1 in updated and isinstance(updated[k1], dict):
            prev_url = str(updated[k1].get("url") or "").strip()

        if not k1 or s1 < 0:
            results.append(MatchResult(
                o.order_number, o.item_id,
                best_score=0.0, second_score=0.0, margin=0.0,
                best_key="", second_key="", third_key="", third_score=0.0,
                action="NO MATCH",
                title=o.title, item_url=o.item_url,
                previous_url=""
            ))
            continue

        if should_auto_update(s1, s2, min_score=min_score, min_margin=min_margin):
            # ensure entry exists
            if k1 not in updated or not isinstance(updated.get(k1), dict):
                updated[k1] = {}

            if prev_url and (not overwrite) and (prev_url != o.item_url):
                action = "SKIP (url exists)"
            else:
                updated[k1]["url"] = o.item_url
                action = "UPDATE" if prev_url != o.item_url else "OK (already set)"
        else:
            action = "NEEDS REVIEW"

        results.append(MatchResult(
            order_number=o.order_number,
            item_id=o.item_id,
            best_score=s1,
            second_score=s2 if s2 >= 0 else 0.0,
            margin=margin if margin >= 0 else 0.0,
            best_key=k1,
            second_key=k2 if k2 else "",
            third_key=k3 if k3 else "",
            third_score=s3 if s3 >= 0 else 0.0,
            action=action,
            title=o.title,
            item_url=o.item_url,
            previous_url=prev_url,
        ))

    return results, updated


# ----------------------------
# Reporting
# ----------------------------

def clip(s: str, n: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else (s[: n - 1] + "…")


def print_table(matches: List[MatchResult]):
    headers = ["order", "item_id", "best", "2nd", "Δ", "best_pdf", "action", "title"]
    rows = []
    for m in matches:
        rows.append([
            m.order_number,
            m.item_id,
            f"{m.best_score:5.1f}",
            f"{m.second_score:5.1f}",
            f"{m.margin:5.1f}",
            clip(m.best_key, 40),
            m.action,
            clip(m.title, 60),
        ])

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    sep = " | "
    line = "-+-".join("-" * w for w in widths)

    print(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(line)
    for r in rows:
        print(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def print_review_details(matches: List[MatchResult], limit: int = 50):
    """
    For NEEDS REVIEW rows, print the top 3 candidates in a readable block format.
    """
    count = 0
    for m in matches:
        if m.action != "NEEDS REVIEW":
            continue
        count += 1
        if count > limit:
            print(f"\n(Review list truncated at {limit} items.)")
            break
        print("\n--- NEEDS REVIEW ---")
        print(f"Order: {m.order_number}  Item: {m.item_id}")
        print(f"Title: {m.title}")
        print(f"URL:   {m.item_url}")
        print("Top candidates:")
        print(f"  1) {m.best_key}  ({m.best_score:.1f})")
        print(f"  2) {m.second_key}  ({m.second_score:.1f})")
        print(f"  3) {m.third_key}  ({m.third_score:.1f})")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Match awaiting-shipment titles to local PDFs and update link JSON (auto-update with margin rule)."
    )
    ap.add_argument("--orders-csv", required=True, help="CSV from awaiting shipment scraper (awaiting_shipment_items.csv).")
    ap.add_argument("--pdf-dir", default=r"c:\Users\benoi\Downloads\ebay_manuals", help="Folder containing PDFs.")
    ap.add_argument("--recursive", action="store_true", help="Scan pdf-dir recursively.")
    ap.add_argument("--links-json", required=True, help="Link DB JSON (pdf base name -> {url}).")
    ap.add_argument("--out-links-json", default="", help="Output path (default overwrites links-json).")
    ap.add_argument("--min-score", type=float, default=60.0, help="Auto-update minimum best score (default: 60).")
    ap.add_argument("--min-margin", type=float, default=8.0, help="Auto-update minimum (best-second) margin (default: 8).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing urls in links JSON.")
    ap.add_argument("--show-review", action="store_true", help="Print expanded details for NEEDS REVIEW rows.")
    args = ap.parse_args()

    orders_csv = Path(args.orders_csv).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    links_json = Path(args.links_json).resolve()
    out_links = Path(args.out_links_json).resolve() if args.out_links_json else links_json

    orders = read_orders_csv(orders_csv)
    pdf_keys = list_pdf_basenames(pdf_dir, recursive=args.recursive)
    if not pdf_keys:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    link_db = load_json(links_json)
    if not isinstance(link_db, dict):
        raise SystemExit(f"links-json must be a JSON object: {links_json}")

    matches, updated = update_links_with_margin(
        orders=orders,
        pdf_keys=pdf_keys,
        link_db=link_db,
        min_score=float(args.min_score),
        min_margin=float(args.min_margin),
        overwrite=bool(args.overwrite),
    )

    print_table(matches)

    upd = sum(1 for m in matches if m.action.startswith("UPDATE"))
    ok = sum(1 for m in matches if m.action.startswith("OK"))
    skip = sum(1 for m in matches if m.action.startswith("SKIP"))
    review = sum(1 for m in matches if m.action == "NEEDS REVIEW")

    print("\nSummary")
    print(f"  PDFs scanned:  {len(pdf_keys)}")
    print(f"  UPDATE:       {upd}")
    print(f"  OK:           {ok}")
    print(f"  SKIP:         {skip}")
    print(f"  NEEDS REVIEW: {review}")
    print(f"  Rule: auto-update if best>= {args.min_score:.1f} and (best-2nd)>= {args.min_margin:.1f}")

    save_json(out_links, updated)
    print(f"\nWrote links JSON: {out_links}")

    if args.show_review and review:
        print_review_details(matches)


if __name__ == "__main__":
    main()

