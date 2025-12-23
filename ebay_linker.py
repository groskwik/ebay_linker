#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from typing import Dict, List, Optional
from urllib.parse import urlparse


# ----------------------------
# Text normalization & scoring
# ----------------------------

_STOPWORDS = {
    "manual", "instruction", "instructions", "owner", "owners", "user", "users", "guide",
    "operation", "service", "schematics", "reference", "advanced", "basic",
    "pages", "page", "with", "and", "&", "for", "the", "a", "an", "of",
    "mk", "mkii", "mkiii", "mark", "series",
}

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^\w\s\-]+", " ", s)
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
    import difflib
    a2, b2 = _norm(a), _norm(b)
    if not a2 and not b2:
        return 1.0
    if not a2 or not b2:
        return 0.0
    return difflib.SequenceMatcher(None, a2, b2).ratio()

def similarity_score(title: str, pdf_base: str) -> float:
    r = _ratio(title, pdf_base)
    j = _jaccard(_tokens(title), _tokens(pdf_base))
    return 100.0 * (0.55 * r + 0.45 * j)


# ----------------------------
# URL -> item_id helper
# ----------------------------

RE_ITM = re.compile(r"/itm/(\d+)")

def extract_item_id_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        path = url or ""
    m = RE_ITM.search(path)
    return m.group(1) if m else ""


# ----------------------------
# PDF inventory
# ----------------------------

@dataclass
class PdfEntry:
    base: str   # filename stem
    path: Path  # full path

def list_pdfs(folder: Path, recursive: bool) -> List[PdfEntry]:
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    paths = list(folder.rglob("*.pdf")) if recursive else list(folder.glob("*.pdf"))

    # Deduplicate by normalized base name, keep first seen
    seen = set()
    out: List[PdfEntry] = []
    for p in paths:
        base = p.stem
        key = _norm(base)
        if key in seen:
            continue
        seen.add(key)
        out.append(PdfEntry(base=base, path=p))

    return out


# ----------------------------
# Links JSON (pdf_base -> {url, item_id, ...})
# ----------------------------

def load_links_json(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("links json must be an object/dict")

    out: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = {str(kk): str(vv) for kk, vv in v.items()}
        else:
            out[k] = {"url": str(v)}
    return out

def save_links_json(path: Path, data: Dict[str, Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved links JSON: {path}")

def build_itemid_index(links: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Build map: item_id -> pdf_base_name
    Uses explicit 'item_id' field if present, else extracts from 'url' if possible.
    """
    idx: Dict[str, str] = {}
    for pdf_base, rec in links.items():
        if not isinstance(rec, dict):
            continue
        item_id = (rec.get("item_id") or "").strip()
        if not item_id:
            item_id = extract_item_id_from_url(rec.get("url", ""))
        if item_id:
            idx[item_id] = pdf_base
    return idx


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
            row.setdefault("item_id", "")
            row.setdefault("title", "")
            row.setdefault("item_url", "")
            rows.append(row)
    return rows


# ----------------------------
# Matching helpers
# ----------------------------

@dataclass
class Candidate:
    pdf: PdfEntry
    score: float

def top_candidates(title: str, pdfs: List[PdfEntry], k: int = 3) -> List[Candidate]:
    scored = [Candidate(p, similarity_score(title, p.base)) for p in pdfs]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]

def choose_match_interactive(
    order_title: str,
    cands: List[Candidate],
    min_score: float,
    min_margin: float,
) -> Optional[PdfEntry]:
    if not cands:
        print("\nNo candidates found.")
        return None

    best = cands[0]
    second = cands[1] if len(cands) > 1 else None
    margin = best.score - (second.score if second else 0.0)
    auto_ok = (best.score >= min_score) and ((second is None) or (margin >= min_margin))

    print("\nOrder title:")
    print(f"  {order_title}")

    print("\nTop matches:")
    for i, c in enumerate(cands, start=1):
        print(f"  {i}. {c.pdf.base}   ({c.score:.1f}%)")

    if auto_ok:
        print(f"\nAuto-selected: {best.pdf.base}  (score={best.score:.1f}%, margin={margin:.1f})")
        return best.pdf

    while True:
        s = input("\nSelect match: 1/2/3, or 0 for no match: ").strip()
        if s == "0":
            return None
        if s in ("1", "2", "3"):
            idx = int(s) - 1
            if idx < len(cands):
                return cands[idx].pdf
            print("That option is not available.")
            continue
        print("Invalid input. Use 1/2/3 or 0.")


# ----------------------------
# myprint automation (NO changes to myprint.py)
# ----------------------------

def find_pdf_matches_like_myprint(pdfs: List[PdfEntry], partial_name: str) -> List[PdfEntry]:
    """
    Emulates your myprint.find_pdf() matching behavior:
    - case-insensitive substring match against filename
    """
    q = (partial_name or "").strip().lower()
    if not q:
        return []
    matches = [p for p in pdfs if q in p.path.name.lower() and p.path.suffix.lower() == ".pdf"]
    return matches

def pick_index_for_exact_basename(matches: List[PdfEntry], chosen_basename: str) -> Optional[int]:
    """
    If myprint would show a numbered list, we can preselect the correct entry by index.
    We match by exact filename (basename + .pdf) if possible.
    Returns 1-based index or None.
    """
    target_pdf = (chosen_basename or "").strip()
    if not target_pdf:
        return None
    target_filename = target_pdf + ".pdf"
    for i, p in enumerate(matches, start=1):
        if p.path.name == target_filename:
            return i
    return None

def run_myprint_with_auto_inputs(
    myprint_path: str,
    python_exe: Optional[str],
    auto_inputs: List[str],
) -> int:
    """
    Run myprint.py and feed ALL prompts via stdin.
    Mirrors your GUI approach: auto_inputs list.
    """
    py = python_exe or sys.executable
    cmd = [py, myprint_path]

    payload = "\n".join(auto_inputs) + "\n"

    print("\n=== Running myprint.py with auto inputs ===")
    print("Command:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

    completed = subprocess.run(cmd, input=payload, text=True)
    return completed.returncode


# ----------------------------
# Main
# ----------------------------

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
                    help="After selecting a PDF, run myprint.py using auto-inputs (no changes to myprint.py).")
    ap.add_argument("--myprint", default="myprint.py",
                    help="Path to myprint.py (default: myprint.py in current directory).")
    ap.add_argument("--python", default=None,
                    help="Python executable to run myprint.py (default: current interpreter).")

    ap.add_argument("--printer", type=str, default="",
                    help="Optional default printer selection (e.g. 1 or 2). If omitted, you will be prompted once.")
    ap.add_argument("--always-ask-printer", action="store_true",
                    help="Ask printer number for every print (default: ask once per run).")

    ap.add_argument("--max-orders", type=int, default=0,
                    help="Optional limit for debugging (0 = no limit).")

    args = ap.parse_args()

    orders = read_orders_csv(args.orders_csv)
    links = load_links_json(args.links_json)
    itemid_index = build_itemid_index(links)

    pdfs = list_pdfs(args.pdf_folder, args.recursive)
    if not pdfs:
        print(f"No PDFs found in: {args.pdf_folder} (recursive={args.recursive})")
        sys.exit(2)

    # Quick lookup for pdf base -> PdfEntry
    pdf_by_normbase: Dict[str, PdfEntry] = {_norm(p.base): p for p in pdfs}

    print(f"Loaded {len(orders)} orders from: {args.orders_csv}")
    print(f"Loaded {len(links)} existing link entries from: {args.links_json}")
    print(f"Indexed {len(pdfs)} unique PDF base names from: {args.pdf_folder} (recursive={args.recursive})")

    default_printer = (args.printer or "").strip()
    if args.do_print and not default_printer and not args.always_ask_printer:
        default_printer = input("\nDefault printer number for this run (e.g. 1 or 2): ").strip()

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
            continue

        chosen_pdf: Optional[PdfEntry] = None

        # --- NEW FEATURE: short-circuit if item_id already known in links JSON ---
        if item_id and item_id in itemid_index:
            known_pdf_base = itemid_index[item_id]
            chosen_pdf = pdf_by_normbase.get(_norm(known_pdf_base))

            print("\nOrder title:")
            print(f"  {title}")
            if chosen_pdf:
                print(f"\nKnown item_id {item_id} already linked to PDF: {chosen_pdf.base} (skipping fuzzy match)")
            else:
                print(
                    f"\nKnown item_id {item_id} is linked to '{known_pdf_base}' in links JSON, "
                    f"but that PDF was not found in the scanned folder. Falling back to fuzzy match."
                )
                chosen_pdf = None

        # Fallback: normal fuzzy matching
        if chosen_pdf is None:
            cands = top_candidates(title, pdfs, k=3)
            chosen_pdf = choose_match_interactive(title, cands, args.min_score, args.min_margin)
            if not chosen_pdf:
                print("No match selected. Moving on.")
                continue

        # Update links JSON: key = PDF base name
        links.setdefault(chosen_pdf.base, {})
        links[chosen_pdf.base]["url"] = url
        if item_id:
            links[chosen_pdf.base]["item_id"] = item_id
            itemid_index[item_id] = chosen_pdf.base  # keep in-run index fresh

        updated += 1
        print(f"Linked: {chosen_pdf.base}  ->  {url}   (item_id={item_id})")

        # Printing workflow
        if args.do_print:
            act = input("Print now? [P]rint / [S]kip / [Q]uit printing: ").strip().lower()
            if act == "":
                act = "p"
            if act.startswith("q"):
                print("Printing disabled for the remainder of this run.")
                args.do_print = False
                continue
            if act.startswith("s"):
                print("Skipped printing; moving to next order.")
                continue

            prn = default_printer
            if args.always_ask_printer or not prn:
                prn = input("Printer number (e.g. 1 or 2): ").strip()

            page_range = input("Page range for myprint (blank = default): ").strip()

            auto_inputs: List[str] = []
            auto_inputs.append(prn)
            auto_inputs.append(chosen_pdf.base)

            matches = find_pdf_matches_like_myprint(pdfs, chosen_pdf.base)
            if len(matches) > 1:
                idx = pick_index_for_exact_basename(matches, chosen_pdf.base)
                if idx is None:
                    idx = 1
                    print("WARNING: multiple PDF matches; exact filename not found. Selecting #1 by default.")
                auto_inputs.append(str(idx))

            auto_inputs.append(page_range)

            rc = run_myprint_with_auto_inputs(args.myprint, args.python, auto_inputs)
            if rc != 0:
                print(f"WARNING: myprint.py returned exit code {rc}. Continuing.")

    save_links_json(args.out_links_json, links)
    print(f"\nDone. Updated/added {updated} links. Processed {processed} order rows.")


if __name__ == "__main__":
    main()

