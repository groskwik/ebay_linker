#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ebay_linker_from_folder_interactive.py

Goal:
- Read eBay orders from a CSV (output of your Selenium scraper).
- Build candidate PDF "base names" directly from a folder of PDFs (optionally recursive).
- Fuzzy-match each order title to PDF names.
- Interactively resolve uncertain matches (choose 1/2/3 or 0 = no match).
- Update a links JSON mapping PDF base name -> {"url": "..."}.
- OPTIONAL: after choosing a PDF, run myprint.py interactively and auto-fill the PDF-hint
  (the "Enter part of the PDF filename:" prompt) with the selected PDF base name.
  You can also skip printing per item.

Notes:
- No pandas.
- Matching uses a robust token-based similarity (pure stdlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------------------
# Text normalization & scoring
# ----------------------------

_STOPWORDS = {
    "manual", "instruction", "instructions", "owner", "owners", "user", "users", "guide",
    "operation", "service", "schematics", "reference", "advanced", "basic", "camera",
    "transceiver", "pages", "page", "with", "and", "&", "for", "the", "a", "an", "of",
    "mk", "mkii", "mkiii", "mark", "series",
}

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^\w\s\-]+", " ", s)   # keep letters/numbers/_ and hyphen
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    s = _norm(s)
    if not s:
        return []
    toks = s.split()
    toks = [t for t in toks if t not in _STOPWORDS and len(t) >= 2]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _ratio(a: str, b: str) -> float:
    # lightweight "similarity ratio" without external libs
    # uses SequenceMatcher on normalized strings
    import difflib
    a2, b2 = _norm(a), _norm(b)
    if not a2 and not b2:
        return 1.0
    if not a2 or not b2:
        return 0.0
    return difflib.SequenceMatcher(None, a2, b2).ratio()

def similarity_score(title: str, pdf_base: str) -> float:
    """
    Score in [0..100].
    Blend:
      - string ratio (SequenceMatcher)
      - token Jaccard overlap
    """
    r = _ratio(title, pdf_base)           # 0..1
    j = _jaccard(_tokens(title), _tokens(pdf_base))  # 0..1
    # weights tuned for your use case (titles are noisy)
    score = 100.0 * (0.55 * r + 0.45 * j)
    return score


# ----------------------------
# PDF inventory
# ----------------------------

def list_pdf_basenames(folder: Path, recursive: bool) -> List[str]:
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {folder}")
    pdfs: List[Path] = []
    if recursive:
        pdfs = list(folder.rglob("*.pdf"))
    else:
        pdfs = list(folder.glob("*.pdf"))
    # base name = filename without extension
    names = [p.stem for p in pdfs]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for n in names:
        key = _norm(n)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


# ----------------------------
# Links JSON
# ----------------------------

def load_links_json(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("links json must be an object/dict")
    # normalize structure: { pdf_base: {"url": "..."} }
    out: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = v
        else:
            out[k] = {"url": str(v)}
    return out

def save_links_json(path: Path, data: Dict[str, Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved links JSON: {path}")


# ----------------------------
# Orders CSV
# ----------------------------

def read_orders_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"orders csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            # Ensure common keys exist
            row.setdefault("item_id", "")
            row.setdefault("title", "")
            row.setdefault("item_url", "")
            rows.append(row)
    return rows


# ----------------------------
# myprint integration
# ----------------------------

def run_myprint_with_prefill(myprint_path: str, pdf_hint: str, python_exe: str | None = None) -> int:
    """
    Run myprint.py interactively BUT inject the PDF hint only when the user indicates
    myprint is at the "Enter part of the PDF filename:" prompt.

    Why this handshake is needed:
    - myprint first asks for printer selection.
    - If we inject too early, the hint can be consumed as the printer number.
    """
    py = python_exe or sys.executable
    cmd = [py, myprint_path]

    print(f"\n=== myprint.py for PDF hint: {pdf_hint} ===")
    print("Launching:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)

    try:
        input("When myprint shows 'Enter part of the PDF filename:', press Enter here to auto-fill it...")
        if p.stdin:
            p.stdin.write(pdf_hint + "\n")
            p.stdin.flush()
            p.stdin.close()
        return p.wait()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating myprint...")
        try:
            p.terminate()
        except Exception:
            pass
        return 130


# ----------------------------
# Matching workflow
# ----------------------------

@dataclass
class Candidate:
    name: str
    score: float

def top_candidates(title: str, pdf_names: List[str], k: int = 3) -> List[Candidate]:
    scored = [Candidate(n, similarity_score(title, n)) for n in pdf_names]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]

def choose_match_interactive(
    order_title: str,
    cands: List[Candidate],
    min_score: float,
    min_margin: float,
) -> str | None:
    """
    Return chosen pdf base name, or None for no match.
    Auto-pick only if:
      - best >= min_score
      - (best - second) >= min_margin  (if second exists)
    Otherwise prompt user to pick 1/2/3 or 0.
    """
    if not cands:
        print("\nNo candidates at all.")
        return None

    best = cands[0]
    second = cands[1] if len(cands) > 1 else None
    margin = best.score - (second.score if second else 0.0)

    auto_ok = (best.score >= min_score) and ((second is None) or (margin >= min_margin))

    print("\nOrder title:")
    print(f"  {order_title}")

    print("\nTop matches:")
    for i, c in enumerate(cands, start=1):
        print(f"  {i}. {c.name}   ({c.score:.1f}%)")

    if auto_ok:
        print(f"\nAuto-selected: {best.name}  (score={best.score:.1f}%, margin={margin:.1f})")
        return best.name

    # interactive
    while True:
        s = input("\nSelect match: 1/2/3, or 0 for no match: ").strip()
        if s == "0":
            return None
        if s in ("1", "2", "3"):
            idx = int(s) - 1
            if idx < len(cands):
                return cands[idx].name
            print("That option is not available.")
            continue
        print("Invalid input. Use 1/2/3 or 0.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders-csv", required=True, type=Path)
    ap.add_argument("--links-json", required=True, type=Path)
    ap.add_argument("--out-links-json", required=True, type=Path)

    ap.add_argument("--pdf-folder", type=Path, default=Path(r"c:\Users\benoi\Downloads\ebay_manuals"),
                    help="Folder containing PDFs (default: c:\\Users\\benoi\\Downloads\\ebay_manuals)")
    ap.add_argument("--recursive", action="store_true", help="Scan PDFs recursively under --pdf-folder")

    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument("--min-margin", type=float, default=8.0)

    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="After selecting a PDF, run myprint.py interactively for that PDF hint.")
    ap.add_argument("--myprint", default="myprint.py",
                    help="Path to myprint.py (default: myprint.py in current directory).")
    ap.add_argument("--python", default=None,
                    help="Python executable to run myprint.py (default: current interpreter).")
    ap.add_argument("--no-print-prompt", action="store_true",
                    help="If set, do not prompt per item; always print when --print is enabled.")
    ap.add_argument("--max-orders", type=int, default=0,
                    help="Optional limit for debugging (0 = no limit).")

    args = ap.parse_args()

    orders = read_orders_csv(args.orders_csv)
    links = load_links_json(args.links_json)

    pdf_names = list_pdf_basenames(args.pdf_folder, args.recursive)
    if not pdf_names:
        print(f"No PDFs found in: {args.pdf_folder} (recursive={args.recursive})")
        sys.exit(2)

    print(f"Loaded {len(orders)} orders from: {args.orders_csv}")
    print(f"Loaded {len(links)} existing link entries from: {args.links_json}")
    print(f"Indexed {len(pdf_names)} unique PDF base names from: {args.pdf_folder} (recursive={args.recursive})")

    updated = 0
    processed = 0

    for row in orders:
        processed += 1
        if args.max_orders and processed > args.max_orders:
            break

        title = (row.get("title") or "").strip()
        url = (row.get("item_url") or "").strip()
        item_id = (row.get("item_id") or "").strip()

        if not title or not url:
            # skip incomplete rows
            continue

        cands = top_candidates(title, pdf_names, k=3)
        chosen = choose_match_interactive(title, cands, args.min_score, args.min_margin)

        if not chosen:
            print("No match selected. Moving on.")
            continue

        # Update links json
        links.setdefault(chosen, {})
        links[chosen]["url"] = url
        updated += 1
        print(f"Linked: {chosen}  ->  {url}   (item_id={item_id})")

        # Optional printing workflow
        if args.do_print:
            if not args.no_print_prompt:
                act = input("Print now? [P]rint / [S]kip / [Q]uit printing: ").strip().lower()
                if act == "":
                    act = "p"
                if act.startswith("q"):
                    print("Printing disabled for the remainder of this run.")
                    args.do_print = False
                elif act.startswith("s"):
                    print("Skipped printing; moving to next order.")
                else:
                    rc = run_myprint_with_prefill(args.myprint, chosen, args.python)
                    if rc != 0:
                        print(f"WARNING: myprint.py returned exit code {rc}. Continuing.")
            else:
                rc = run_myprint_with_prefill(args.myprint, chosen, args.python)
                if rc != 0:
                    print(f"WARNING: myprint.py returned exit code {rc}. Continuing.")

    save_links_json(args.out_links_json, links)
    print(f"\nDone. Updated/added {updated} links. Processed {processed} order rows.")


if __name__ == "__main__":
    main()

