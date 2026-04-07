#!/usr/bin/env python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
match_manuals.py

Interactive helper to normalize manuals.csv "title" field to the selected PDF basename.

- Reads CSV row by row
- For each row, finds best matching PDF filenames in two folders
- User picks candidate by number (or skip)
- Updates only the "title" column to the chosen PDF stem (basename without .pdf)
- Writes manuals.updated.csv and creates a backup manuals.csv.bak

Usage:
  python match_manuals.py --csv "C:\\path\\manuals.csv"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PDF_DIRS_DEFAULT = [
    r"C:\Users\benoi\Downloads\ebay_manuals",
    r"C:\Users\benoi\Downloads\Manuals",
]

# -------------------- matching helpers --------------------

_token_re = re.compile(r"[A-Za-z0-9]+")

def normalize_for_match(s: str) -> str:
    """Normalize string for matching: lowercase, keep alnum tokens, collapse spaces."""
    s = (s or "").lower()
    tokens = _token_re.findall(s)
    return " ".join(tokens)

def score(a_norm: str, b_norm: str) -> float:
    """Return similarity score in [0, 100]."""
    if not a_norm and not b_norm:
        return 100.0
    if not a_norm or not b_norm:
        return 0.0
    return 100.0 * SequenceMatcher(None, a_norm, b_norm).ratio()

@dataclass(frozen=True)
class PdfEntry:
    stem: str              # filename without extension
    path: Path             # full path
    norm: str              # normalized stem

def build_pdf_index(pdf_dirs: List[str]) -> List[PdfEntry]:
    entries: List[PdfEntry] = []
    seen: set[Tuple[str, str]] = set()

    for d in pdf_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] PDF folder not found: {p}")
            continue

        for pdf in p.rglob("*.pdf"):
            try:
                stem = pdf.stem
            except Exception:
                continue
            key = (stem.lower(), str(pdf).lower())
            if key in seen:
                continue
            seen.add(key)
            entries.append(PdfEntry(stem=stem, path=pdf, norm=normalize_for_match(stem)))

    # stable sort for repeatability
    entries.sort(key=lambda e: (e.stem.lower(), str(e.path).lower()))
    return entries

def top_matches(title: str, pdf_entries: List[PdfEntry], k: int = 12) -> List[Tuple[float, PdfEntry]]:
    tnorm = normalize_for_match(title)
    scored: List[Tuple[float, PdfEntry]] = [(score(tnorm, e.norm), e) for e in pdf_entries]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

# -------------------- I/O helpers --------------------

def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return fieldnames, rows

def write_csv_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def prompt_choice(matches: List[Tuple[float, PdfEntry]]) -> Optional[PdfEntry]:
    """
    Prompt user:
      - number selects that PDF
      - 's' skip
      - 'q' quit (returns None and signals quit by raising SystemExit)
    """
    while True:
        raw = input("Choose [1..N], (s)kip, (q)uit: ").strip().lower()
        if raw == "q":
            raise SystemExit(0)
        if raw == "s" or raw == "":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(matches):
                return matches[idx - 1][1]
        print("Invalid choice. Enter a number, 's', or 'q'.")

# -------------------- main --------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to manuals.csv")
    ap.add_argument("--pdf-dir", action="append", default=None,
                    help="PDF folder to search (can be repeated). Defaults to the 2 standard folders.")
    ap.add_argument("--top", type=int, default=30, help="How many matches to show per row.")
    ap.add_argument("--min-score", type=float, default=0.0, help="Hide candidates below this score (0-100).")
    ap.add_argument("--auto-exact", action="store_true",
                    help="If title already exactly equals a PDF stem (case-insensitive), skip prompting.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 2

    pdf_dirs = args.pdf_dir if args.pdf_dir else PDF_DIRS_DEFAULT
    pdf_entries = build_pdf_index(pdf_dirs)
    if not pdf_entries:
        print("[ERROR] No PDFs found in the provided folders.")
        return 3

    # Map for exact stem matching (case-insensitive)
    stem_map: Dict[str, PdfEntry] = {e.stem.lower(): e for e in pdf_entries}

    fieldnames, rows = read_csv_rows(csv_path)
    if "title" not in fieldnames:
        print(f"[ERROR] CSV must contain a 'title' column. Found: {fieldnames}")
        return 4

    # Backup original
    bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(csv_path, bak_path)
        print(f"[OK] Backup created: {bak_path}")
    else:
        print(f"[INFO] Backup already exists: {bak_path}")

    updated = 0

    for i, row in enumerate(rows, start=1):
        old_title = (row.get("title") or "").strip()
        if not old_title:
            print(f"\n[{i}/{len(rows)}] title is empty -> skip")
            continue

        # Optional: if already exact match, skip prompting
        if args.auto_exact:
            exact = stem_map.get(old_title.lower())
            if exact:
                # Normalize to exact casing of file stem
                if row["title"] != exact.stem:
                    row["title"] = exact.stem
                    updated += 1
                print(f"\n[{i}/{len(rows)}] '{old_title}' already matches PDF stem -> keep '{row['title']}'")
                continue

        print(f"\n[{i}/{len(rows)}] CSV title: {old_title}")

        matches = top_matches(old_title, pdf_entries, k=max(1, args.top))
        # filter by min-score
        matches = [(sc, e) for (sc, e) in matches if sc >= args.min_score]

        if not matches:
            print("  No candidates above min-score. (s)kip or (q)uit.")
            try:
                _ = prompt_choice([])
            except SystemExit:
                print("[INFO] Quit.")
                break
            continue

        for j, (sc, e) in enumerate(matches, start=1):
            # show stem and relative folder for clarity
            folder = str(e.path.parent)
            print(f"  {j:2d}) {sc:5.1f}  {e.stem}   [{folder}]")

        try:
            chosen = prompt_choice(matches)
        except SystemExit:
            print("[INFO] Quit.")
            break

        if chosen is None:
            print("  -> skipped (title unchanged)")
            continue

        new_title = chosen.stem
        if new_title != old_title:
            row["title"] = new_title
            updated += 1
            print(f"  -> UPDATED title to: {new_title}")
        else:
            print("  -> unchanged")

    out_path = csv_path.with_name(csv_path.stem + ".updated" + csv_path.suffix)
    write_csv_rows(out_path, fieldnames, rows)
    print(f"\n[OK] Wrote: {out_path}")
    print(f"[OK] Updated {updated} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
