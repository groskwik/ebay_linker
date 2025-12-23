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
    return [p.stem for p in pdfs if p.is_file()]


# ----------------------------
# Core logic
# ----------------------------

def should_auto_update(best: float, second: float, min_score: float, min_margin: float) -> bool:
    return (best >= min_score) and ((best - second) >= min_margin)


def ensure_entry(link_db: Dict[str, dict], key: str) -> None:
    if key not in link_db or not isinstance(link_db.get(key), dict):
        link_db[key] = {}


def clip(s: str, n: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else (s[: n - 1] + "…")


def prompt_choice(order: OrderRow, top3: List[Tuple[str, float]]) -> str:
    """
    Returns one of: "0","1","2","3","s","q"
    """
    (k1, s1), (k2, s2), (k3, s3) = top3
    print("\n--- NEEDS INPUT ---")
    print(f"Order: {order.order_number}   Item: {order.item_id}")
    print(f"Title: {order.title}")
    print(f"URL:   {order.item_url}")
    print("\nCandidates:")
    print(f"  1) {k1}  ({s1:.1f})")
    print(f"  2) {k2}  ({s2:.1f})")
    print(f"  3) {k3}  ({s3:.1f})")
    print("\nEnter choice: 1/2/3 to select, 0 = no match, s = skip, q = quit")
    while True:
        ans = input("> ").strip().lower()
        if ans in {"0", "1", "2", "3", "s", "q"}:
            return ans
        print("Invalid input. Please enter 0,1,2,3,s, or q.")


def process_orders(
    orders: List[OrderRow],
    pdf_keys: List[str],
    link_db: Dict[str, dict],
    min_score: float,
    min_margin: float,
    overwrite: bool,
    interactive: bool,
) -> Dict[str, dict]:
    updated = dict(link_db)

    for o in orders:
        top3 = top_n_matches(o.title, pdf_keys, n=3)
        while len(top3) < 3:
            top3.append(("", -1.0))

        (k1, s1), (k2, s2), (k3, s3) = top3
        s2_eff = s2 if s2 >= 0 else 0.0

        auto_ok = k1 and should_auto_update(s1, s2_eff, min_score=min_score, min_margin=min_margin)

        # Determine which key we will use, if any
        chosen_key = ""
        chosen_score = 0.0

        if auto_ok:
            chosen_key, chosen_score = k1, s1
        elif interactive:
            choice = prompt_choice(o, top3)
            if choice == "q":
                print("Quitting early (progress will be saved).")
                break
            if choice == "s" or choice == "0":
                # skip / no match
                continue
            idx = int(choice) - 1
            chosen_key, chosen_score = top3[idx]
            if not chosen_key:
                # defensive
                continue
        else:
            # non-interactive: skip ambiguous ones
            continue

        # Apply update for chosen_key
        ensure_entry(updated, chosen_key)
        prev_url = str(updated[chosen_key].get("url") or "").strip()

        if prev_url and (not overwrite) and (prev_url != o.item_url):
            # keep existing
            continue

        updated[chosen_key]["url"] = o.item_url

    return updated


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Match awaiting-shipment titles to local PDFs and update link JSON (interactive selection for unknowns)."
    )
    ap.add_argument("--orders-csv", required=True, help="CSV from awaiting shipment scraper (awaiting_shipment_items.csv).")
    ap.add_argument("--pdf-dir", default=r"c:\Users\benoi\Downloads\ebay_manuals", help="Folder containing PDFs.")
    ap.add_argument("--recursive", action="store_true", help="Scan pdf-dir recursively.")
    ap.add_argument("--links-json", required=True, help="Link DB JSON (pdf base name -> {url}).")
    ap.add_argument("--out-links-json", default="", help="Output path (default overwrites links-json).")
    ap.add_argument("--min-score", type=float, default=60.0, help="Auto-update minimum best score (default: 60).")
    ap.add_argument("--min-margin", type=float, default=8.0, help="Auto-update minimum (best-second) margin (default: 8).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing urls in links JSON.")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt; skip ambiguous matches.")
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

    updated = process_orders(
        orders=orders,
        pdf_keys=pdf_keys,
        link_db=link_db,
        min_score=float(args.min_score),
        min_margin=float(args.min_margin),
        overwrite=bool(args.overwrite),
        interactive=(not args.non_interactive),
    )

    save_json(out_links, updated)
    print(f"\nWrote links JSON: {out_links}")


if __name__ == "__main__":
    main()

