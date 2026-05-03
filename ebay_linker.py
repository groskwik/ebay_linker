#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass


# =============================================================================
# Normalization / similarity
# =============================================================================

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "manual", "user", "users", "guide", "instruction", "instructions",
    "reference", "owner", "owners", "operating", "operation",
}


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
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


def similarity_score(title: str, other: str) -> float:
    r = _ratio(title, other)
    j = _jaccard(_tokens(title), _tokens(other))
    return 100.0 * (0.55 * r + 0.45 * j)


# =============================================================================
# URL -> item_id helper
# =============================================================================

RE_ITM = re.compile(r"/itm/(\d+)")


def extract_item_id_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        path = url or ""
    m = RE_ITM.search(path)
    return m.group(1) if m else ""


# =============================================================================
# PDF inventory
# =============================================================================

@dataclass
class PdfEntry:
    base: str
    path: Path


@dataclass
class InventoryHit:
    title: str
    box: Optional[str]
    cover: str


@dataclass
class SkippedInventoryRecord:
    pdf_base: str
    pdf_path: str
    location: str


class InventorySkipCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: Dict[str, SkippedInventoryRecord] = {}

    def add(self, record: SkippedInventoryRecord) -> None:
        key = _norm(record.pdf_base)
        with self._lock:
            self._records[key] = record

    def has_records(self) -> bool:
        with self._lock:
            return bool(self._records)

    def records(self) -> List[SkippedInventoryRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: _norm(r.pdf_base))


# =============================================================================
# manuals.csv inventory lookup (mirrors myprint.py logic)
# =============================================================================


def normalize_for_db(s: str) -> str:
    s = (s or "").lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    return " ".join(tokens)


DEFAULT_MANUALS_CSV = Path(r"C:\Users\benoi\Downloads\ManualForge\manuals.csv")


class ManualsInventory:
    def __init__(self, by_title: Dict[str, List[InventoryHit]], csv_path: Path) -> None:
        self.by_title = by_title
        self.csv_path = csv_path

    @classmethod
    def load(cls, csv_path: Path) -> "ManualsInventory":
        if not csv_path.exists():
            print(f"[inventory] manuals.csv not found at: {csv_path} (inventory check disabled)")
            return cls({}, csv_path)

        by_title: Dict[str, List[InventoryHit]] = {}
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = (row.get("title") or "").strip()
                    ntitle = normalize_for_db(title)
                    if not ntitle:
                        continue
                    hit = InventoryHit(
                        title=title,
                        box=((row.get("box") or "").strip() or None),
                        cover=(row.get("cover") or "").strip(),
                    )
                    by_title.setdefault(ntitle, []).append(hit)
        except Exception as e:
            print(f"[inventory] WARNING: failed to read manuals.csv {csv_path}: {e}")
            return cls({}, csv_path)

        return cls(by_title, csv_path)

    def lookup_pdf(self, pdf: PdfEntry) -> List[InventoryHit]:
        return list(self.by_title.get(normalize_for_db(pdf.base), []))


def interpret_inventory_hit(hit: InventoryHit) -> str:
    parts: List[str] = []
    if hit.box:
        parts.append(f"in {hit.box}")
    if hit.cover == "1":
        parts.append("cover-only (cover=1)")
    elif hit.cover == "0":
        parts.append("not cover-only (cover=0)")
    elif hit.cover:
        parts.append(f"cover={hit.cover}")

    if not parts:
        return "present (no box/cover info)"
    return ", ".join(parts)


def inventory_location_summary(hits: List[InventoryHit]) -> str:
    if not hits:
        return ""
    return " | ".join(interpret_inventory_hit(h) for h in hits)


def list_pdfs(folder: Path, recursive: bool) -> List[PdfEntry]:
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    paths = list(folder.rglob("*.pdf")) if recursive else list(folder.glob("*.pdf"))

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


def get_pdf_pagecount(pdf_path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            print("WARNING: Neither 'pypdf' nor 'PyPDF2' is installed; cannot read page counts.")
            return None

    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as e:
        print(f"WARNING: failed to read page count for {pdf_path}: {e}")
        return None


# =============================================================================
# Links JSON (pdf_base -> {url, item_id, ...})
# =============================================================================


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


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def build_itemid_index_and_flags(links: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, bool]]:
    idx: Dict[str, str] = {}
    tw: Dict[str, bool] = {}

    for pdf_base, rec in links.items():
        if not isinstance(rec, dict):
            continue

        item_id = (rec.get("item_id") or "").strip()
        if not item_id:
            item_id = extract_item_id_from_url(rec.get("url", ""))
        if not item_id:
            continue

        idx[item_id] = pdf_base
        tw[item_id] = _as_bool(rec.get("typewriter", False))

    return idx, tw


# =============================================================================
# Orders CSV
# =============================================================================


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


# =============================================================================
# Printed manual inventory CSV (optional) - kept for compatibility
# =============================================================================

@dataclass
class ManualEntry:
    title: str
    box: Optional[str]
    cover: bool


def load_manuals_from_csv_any(path: Path) -> Dict[str, ManualEntry]:
    out: Dict[str, ManualEntry] = {}
    if not path or not path.exists():
        return out

    first_line = ""
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if line.strip():
                first_line = line.strip()
                break
    if not first_line:
        return out

    lower = first_line.lower()
    has_header = ("title" in lower and "box" in lower and "cover" in lower)

    if has_header:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                if not title:
                    continue
                box_raw = (row.get("box") or "").strip()
                box = box_raw or None
                cover_raw = (row.get("cover") or "").strip().lower()
                cover = cover_raw in ("1", "true", "yes", "y", "on")
                out[title] = ManualEntry(title=title, box=box, cover=cover)
        return out

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                title = row[0].strip()
                box = None
                cover = False
            elif len(row) == 2:
                title = row[0].strip()
                box = row[1].strip() or None
                cover = False
            else:
                title = ",".join(row[:-2]).strip()
                box = row[-2].strip() or None
                cover_raw = (row[-1] or "").strip().lower()
                cover = cover_raw in ("1", "true", "yes", "y", "on")

            if not title:
                continue
            out[title] = ManualEntry(title=title, box=box, cover=cover)
    return out


# =============================================================================
# Matching helpers (normal mode only)
# =============================================================================

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


# =============================================================================
# myprint automation (with inventory-aware skip handling)
# =============================================================================


def find_pdf_matches_like_myprint(pdfs: List[PdfEntry], partial_name: str) -> List[PdfEntry]:
    q = (partial_name or "").strip().lower()
    if not q:
        return []
    matches = [p for p in pdfs if q in p.path.name.lower() and p.path.suffix.lower() == ".pdf"]
    return matches


def pick_index_for_exact_basename(matches: List[PdfEntry], chosen_basename: str) -> Optional[int]:
    target_pdf = (chosen_basename or "").strip()
    if not target_pdf:
        return None
    target_filename = target_pdf + ".pdf"
    for i, p in enumerate(matches, start=1):
        if p.path.name == target_filename:
            return i
    return None


@dataclass
class MyPrintResult:
    exit_code: int
    skipped_in_inventory: bool = False
    inventory_location: str = ""


def run_myprint_with_auto_inputs(myprint_path: str, python_exe: Optional[str], auto_inputs: List[str]) -> int:
    py = python_exe or sys.executable
    cmd = [py, myprint_path]
    payload = "\n".join(auto_inputs) + "\n"

    print("\n=== Running myprint.py with auto inputs ===")
    print("Command:", " ".join(f'\"{c}\"' if " " in c else c for c in cmd))
    print("Auto-inputs:", auto_inputs)

    completed = subprocess.run(cmd, input=payload, text=True)
    return completed.returncode


def myprint_auto_print_range(
    *,
    pdfs: List[PdfEntry],
    chosen_pdf: PdfEntry,
    printer: str,
    page_range: str,
    myprint_path: str,
    python_exe: Optional[str],
    inventory: Optional[ManualsInventory] = None,
    skip_collector: Optional[InventorySkipCollector] = None,
) -> MyPrintResult:
    hits: List[InventoryHit] = []
    if inventory is not None:
        hits = inventory.lookup_pdf(chosen_pdf)

    if hits:
        location = inventory_location_summary(hits)
        print(f"\n[inventory] SKIP printing '{chosen_pdf.base}' because it is already in manuals.csv")
        print(f"[inventory] Location/info: {location}")
        if skip_collector is not None:
            skip_collector.add(
                SkippedInventoryRecord(
                    pdf_base=chosen_pdf.base,
                    pdf_path=str(chosen_pdf.path),
                    location=location,
                )
            )
        return MyPrintResult(exit_code=0, skipped_in_inventory=True, inventory_location=location)

    auto_inputs: List[str] = []
    auto_inputs.append(printer)
    auto_inputs.append(chosen_pdf.base)

    matches = find_pdf_matches_like_myprint(pdfs, chosen_pdf.base)
    if len(matches) > 1:
        idx = pick_index_for_exact_basename(matches, chosen_pdf.base)
        if idx is None:
            idx = 1
            print("WARNING: multiple PDF matches; exact filename not found. Selecting #1 by default.")
        auto_inputs.append(str(idx))

    auto_inputs.append(page_range)
    rc = run_myprint_with_auto_inputs(myprint_path, python_exe, auto_inputs)
    return MyPrintResult(exit_code=rc)


# =============================================================================
# print360 mode (inventory-aware)
# =============================================================================

@dataclass
class Print360Resume:
    order_index: int
    pdf: PdfEntry
    total_pages: int
    next_page: int


def run_print360_batch(
    *,
    orders: List[Dict[str, str]],
    start_index: int,
    start_resume: Optional[Print360Resume] = None,
    itemid_index: Dict[str, str],
    pdf_by_normbase: Dict[str, PdfEntry],
    pdfs: List[PdfEntry],
    printer: str,
    myprint_path: str,
    python_exe: Optional[str],
    page_limit: int = 360,
    inventory: Optional[ManualsInventory] = None,
    skip_collector: Optional[InventorySkipCollector] = None,
) -> Tuple[int, int, Optional[Print360Resume]]:
    """Print one print360 batch and return the next position.

    The batch stops when approximately ``page_limit`` pages have been printed.
    If a manual is split across batches, ``resume`` points to the same order and
    the next page to print. The caller can pass that resume back into this
    function to continue the next 360-page batch without switching to normal
    interactive mode.
    """
    pages_printed = 0
    idx = start_index
    resume: Optional[Print360Resume] = None

    while idx < len(orders) and pages_printed < page_limit:
        row = orders[idx]
        title = (row.get("title") or "").strip()
        item_id = (row.get("item_id") or "").strip()
        url = (row.get("item_url") or "").strip()

        if not title or not url:
            idx += 1
            start_resume = None
            continue

        if not item_id or item_id not in itemid_index:
            print(f"\n[print360] SKIP (not in links DB): item_id={item_id!r}  title={title}")
            idx += 1
            start_resume = None
            continue

        pdf_base = itemid_index[item_id]
        pdf = pdf_by_normbase.get(_norm(pdf_base))
        if not pdf:
            print(f"\n[print360] SKIP (PDF not found): item_id={item_id}  pdf_base={pdf_base!r}")
            idx += 1
            start_resume = None
            continue

        total_pages = get_pdf_pagecount(pdf.path)
        if not total_pages or total_pages <= 0:
            print(f"\n[print360] SKIP (cannot read page count): {pdf.path}")
            idx += 1
            start_resume = None
            continue

        start_page = 1
        if start_resume is not None and start_resume.order_index == idx:
            start_page = max(1, min(start_resume.next_page, total_pages + 1))

        if start_page > total_pages:
            idx += 1
            start_resume = None
            continue

        remaining_capacity = page_limit - pages_printed
        if remaining_capacity <= 0:
            break

        remaining_pages = total_pages - start_page + 1

        if remaining_pages <= remaining_capacity:
            end_page = total_pages
            pr = f"{start_page}-{end_page}"
            print(f"\n[print360] PRINT: {pdf.base}  pages={pr}  (total={total_pages})")
            result = myprint_auto_print_range(
                pdfs=pdfs,
                chosen_pdf=pdf,
                printer=printer,
                page_range=pr,
                myprint_path=myprint_path,
                python_exe=python_exe,
                inventory=inventory,
                skip_collector=skip_collector,
            )
            if result.exit_code != 0:
                print(f"[print360] WARNING: myprint exit code {result.exit_code} (continuing)")
            elif not result.skipped_in_inventory:
                pages_printed += remaining_pages
            idx += 1
            start_resume = None
            continue

        end_page = start_page + remaining_capacity - 1
        if end_page % 2 == 1:
            end_page += 1

        if end_page > total_pages:
            end_page = total_pages
            if end_page % 2 == 1 and end_page - 1 >= start_page:
                end_page -= 1

        pr = f"{start_page}-{end_page}"
        printed_now = (end_page - start_page + 1) if end_page >= start_page else 0

        print(f"\n[print360] PRINT PARTIAL (even-ended): {pdf.base}  pages={pr}  (total={total_pages})")
        result = myprint_auto_print_range(
            pdfs=pdfs,
            chosen_pdf=pdf,
            printer=printer,
            page_range=pr,
            myprint_path=myprint_path,
            python_exe=python_exe,
            inventory=inventory,
            skip_collector=skip_collector,
        )
        if result.exit_code != 0:
            print(f"[print360] WARNING: myprint exit code {result.exit_code} (continuing)")
            resume = Print360Resume(order_index=idx, pdf=pdf, total_pages=total_pages, next_page=start_page)
            break
        elif not result.skipped_in_inventory:
            pages_printed += printed_now
            if end_page < total_pages:
                resume = Print360Resume(order_index=idx, pdf=pdf, total_pages=total_pages, next_page=end_page + 1)
                break

        idx += 1
        start_resume = None

    return idx, pages_printed, resume

def finish_resume_manual(
    *,
    resume: Print360Resume,
    printer: str,
    pdfs: List[PdfEntry],
    myprint_path: str,
    python_exe: Optional[str],
    inventory: Optional[ManualsInventory] = None,
    skip_collector: Optional[InventorySkipCollector] = None,
) -> None:
    if resume.next_page > resume.total_pages:
        return
    pr = f"{resume.next_page}-{resume.total_pages}"
    print(f"\n[resume] FINISH MANUAL: {resume.pdf.base}  pages={pr}  (total={resume.total_pages})")
    result = myprint_auto_print_range(
        pdfs=pdfs,
        chosen_pdf=resume.pdf,
        printer=printer,
        page_range=pr,
        myprint_path=myprint_path,
        python_exe=python_exe,
        inventory=inventory,
        skip_collector=skip_collector,
    )
    if result.exit_code != 0:
        print(f"[resume] WARNING: myprint exit code {result.exit_code} (continuing)")


# =============================================================================
# print720 mode (typewriter-aware, dry run split + concurrent execution + persistent state)
# =============================================================================

@dataclass
class EligibleDoc:
    order_index: int
    pdf: PdfEntry
    total_pages: int


@dataclass
class PrintTask:
    pdf: PdfEntry
    start_page: int
    end_page: int

    @property
    def page_range(self) -> str:
        return f"{self.start_page}-{self.end_page}"

    @property
    def pages(self) -> int:
        return max(0, self.end_page - self.start_page + 1)


@dataclass
class PrintStreamPos:
    doc_list_index: int
    next_page: int


@dataclass
class Print720Plan:
    tasks_p1: List[PrintTask]
    tasks_p2: List[PrintTask]
    printed_p1: int
    printed_p2: int
    end_pos_normal: PrintStreamPos
    end_pos_typewriter: PrintStreamPos
    has_more_normal: bool
    has_more_typewriter: bool


@dataclass
class Print720State:
    normal_doc_list_index: int = 0
    normal_next_page: int = 1
    typewriter_doc_list_index: int = 0
    typewriter_next_page: int = 1

    def normal_pos(self) -> PrintStreamPos:
        return PrintStreamPos(self.normal_doc_list_index, self.normal_next_page)

    def typewriter_pos(self) -> PrintStreamPos:
        return PrintStreamPos(self.typewriter_doc_list_index, self.typewriter_next_page)

    @classmethod
    def from_positions(cls, normal: PrintStreamPos, typewriter: PrintStreamPos) -> "Print720State":
        return cls(
            normal_doc_list_index=normal.doc_list_index,
            normal_next_page=normal.next_page,
            typewriter_doc_list_index=typewriter.doc_list_index,
            typewriter_next_page=typewriter.next_page,
        )


def load_print720_state(path: Path) -> Print720State:
    if not path.exists():
        return Print720State()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return Print720State()
        return Print720State(
            normal_doc_list_index=int(data.get("normal_doc_list_index", 0)),
            normal_next_page=int(data.get("normal_next_page", 1)),
            typewriter_doc_list_index=int(data.get("typewriter_doc_list_index", 0)),
            typewriter_next_page=int(data.get("typewriter_next_page", 1)),
        )
    except Exception as e:
        print(f"[print720] WARNING: failed to load state file {path}: {e}")
        return Print720State()


def save_print720_state(path: Path, state: Print720State) -> None:
    try:
        payload = {
            "normal_doc_list_index": int(state.normal_doc_list_index),
            "normal_next_page": int(state.normal_next_page),
            "typewriter_doc_list_index": int(state.typewriter_doc_list_index),
            "typewriter_next_page": int(state.typewriter_next_page),
            "saved_at": int(time.time()),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[print720] Saved state: {path}")
    except Exception as e:
        print(f"[print720] WARNING: failed to save state file {path}: {e}")


def reset_print720_state(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            print(f"[print720] Reset state (deleted): {path}")
    except Exception as e:
        print(f"[print720] WARNING: failed to delete state file {path}: {e}")


def _build_eligible_docs(
    *,
    orders: List[Dict[str, str]],
    itemid_index: Dict[str, str],
    itemid_typewriter: Dict[str, bool],
    pdf_by_normbase: Dict[str, PdfEntry],
    want_typewriter: bool,
) -> List[EligibleDoc]:
    out: List[EligibleDoc] = []
    for i, row in enumerate(orders):
        title = (row.get("title") or "").strip()
        url = (row.get("item_url") or "").strip()
        item_id = (row.get("item_id") or "").strip()

        if not title or not url:
            continue
        if not item_id or item_id not in itemid_index:
            continue

        is_tw = bool(itemid_typewriter.get(item_id, False))
        if want_typewriter != is_tw:
            continue

        pdf_base = itemid_index[item_id]
        pdf = pdf_by_normbase.get(_norm(pdf_base))
        if not pdf:
            continue

        total_pages = get_pdf_pagecount(pdf.path)
        if not total_pages or total_pages <= 0:
            continue

        out.append(EligibleDoc(order_index=i, pdf=pdf, total_pages=total_pages))
    return out


def _advance_pos(docs: List[EligibleDoc], pos: PrintStreamPos) -> PrintStreamPos:
    while pos.doc_list_index < len(docs):
        d = docs[pos.doc_list_index]
        if pos.next_page <= d.total_pages:
            return pos
        pos = PrintStreamPos(doc_list_index=pos.doc_list_index + 1, next_page=1)
    return pos


def _plan_for_one_printer(
    *,
    docs: List[EligibleDoc],
    start_pos: PrintStreamPos,
    page_limit: int,
    force_even_end_if_cut: bool = True,
) -> Tuple[List[PrintTask], int, PrintStreamPos]:
    tasks: List[PrintTask] = []
    allocated = 0
    pos = _advance_pos(docs, start_pos)

    while pos.doc_list_index < len(docs) and allocated < page_limit:
        d = docs[pos.doc_list_index]
        start_page = pos.next_page
        remaining_in_doc = d.total_pages - start_page + 1
        remaining_capacity = page_limit - allocated

        if remaining_in_doc <= remaining_capacity:
            end_page = d.total_pages
            tasks.append(PrintTask(pdf=d.pdf, start_page=start_page, end_page=end_page))
            allocated += (end_page - start_page + 1)
            pos = PrintStreamPos(doc_list_index=pos.doc_list_index + 1, next_page=1)
            pos = _advance_pos(docs, pos)
            continue

        end_page = start_page + remaining_capacity - 1
        if force_even_end_if_cut and (end_page % 2 == 1):
            end_page += 1

        if end_page > d.total_pages:
            end_page = d.total_pages
            if force_even_end_if_cut and (end_page % 2 == 1) and (end_page - 1 >= start_page):
                end_page -= 1

        tasks.append(PrintTask(pdf=d.pdf, start_page=start_page, end_page=end_page))
        allocated += max(0, end_page - start_page + 1)
        pos = PrintStreamPos(doc_list_index=pos.doc_list_index, next_page=end_page + 1)
        pos = _advance_pos(docs, pos)
        break

    return tasks, allocated, pos


def plan_print720(
    *,
    orders: List[Dict[str, str]],
    itemid_index: Dict[str, str],
    itemid_typewriter: Dict[str, bool],
    pdf_by_normbase: Dict[str, PdfEntry],
    start_pos_normal: PrintStreamPos,
    start_pos_typewriter: PrintStreamPos,
    limit_each: int = 360,
    typewriter_printer: int = 0,
) -> Print720Plan:
    docs_normal = _build_eligible_docs(
        orders=orders,
        itemid_index=itemid_index,
        itemid_typewriter=itemid_typewriter,
        pdf_by_normbase=pdf_by_normbase,
        want_typewriter=False,
    )
    docs_tw = _build_eligible_docs(
        orders=orders,
        itemid_index=itemid_index,
        itemid_typewriter=itemid_typewriter,
        pdf_by_normbase=pdf_by_normbase,
        want_typewriter=True,
    )

    posN0 = _advance_pos(docs_normal, start_pos_normal)
    posT0 = _advance_pos(docs_tw, start_pos_typewriter)

    if typewriter_printer == 1:
        t1, p1, posT1 = _plan_for_one_printer(docs=docs_tw, start_pos=posT0, page_limit=limit_each, force_even_end_if_cut=True)
        t2, p2, posN1 = _plan_for_one_printer(docs=docs_normal, start_pos=posN0, page_limit=limit_each, force_even_end_if_cut=True)
        endN = posN1
        endT = posT1
    elif typewriter_printer == 2:
        t1, p1, posN1 = _plan_for_one_printer(docs=docs_normal, start_pos=posN0, page_limit=limit_each, force_even_end_if_cut=True)
        t2, p2, posT1 = _plan_for_one_printer(docs=docs_tw, start_pos=posT0, page_limit=limit_each, force_even_end_if_cut=True)
        endN = posN1
        endT = posT1
    else:
        t1, p1, posN1 = _plan_for_one_printer(docs=docs_normal, start_pos=posN0, page_limit=limit_each, force_even_end_if_cut=True)
        t2, p2, posN2 = _plan_for_one_printer(docs=docs_normal, start_pos=posN1, page_limit=limit_each, force_even_end_if_cut=True)
        endN = posN2
        endT = posT0

    endN = _advance_pos(docs_normal, endN)
    endT = _advance_pos(docs_tw, endT)

    has_more_normal = endN.doc_list_index < len(docs_normal)
    has_more_typewriter = endT.doc_list_index < len(docs_tw)

    return Print720Plan(
        tasks_p1=t1,
        tasks_p2=t2,
        printed_p1=p1,
        printed_p2=p2,
        end_pos_normal=endN,
        end_pos_typewriter=endT,
        has_more_normal=has_more_normal,
        has_more_typewriter=has_more_typewriter,
    )


def _print_plan_summary(plan: Print720Plan, typewriter_printer: int) -> None:
    tw_note = {0: "none", 1: "printer1", 2: "printer2"}.get(typewriter_printer, "none")
    print("\n[print720] Dry run plan:")
    print(f"  Typewriter printer: {tw_note}")
    print(f"  Printer 1 pages: {plan.printed_p1} (target ~360, may be 361 for duplex)")
    for t in plan.tasks_p1:
        print(f"    - {t.pdf.base}: {t.page_range}  ({t.pages} pages)")
    print(f"  Printer 2 pages: {plan.printed_p2} (target ~360, may be 361 for duplex)")
    for t in plan.tasks_p2:
        print(f"    - {t.pdf.base}: {t.page_range}  ({t.pages} pages)")


def _run_tasks_for_printer(
    *,
    printer: str,
    tasks: List[PrintTask],
    pdfs: List[PdfEntry],
    myprint_path: str,
    python_exe: Optional[str],
    tag: str,
    inventory: Optional[ManualsInventory] = None,
    skip_collector: Optional[InventorySkipCollector] = None,
) -> int:
    last_rc = 0
    for t in tasks:
        print(f"\n[{tag}] PRINT: {t.pdf.base}  pages={t.page_range}  on printer={printer}")
        result = myprint_auto_print_range(
            pdfs=pdfs,
            chosen_pdf=t.pdf,
            printer=printer,
            page_range=t.page_range,
            myprint_path=myprint_path,
            python_exe=python_exe,
            inventory=inventory,
            skip_collector=skip_collector,
        )
        if result.exit_code != 0:
            last_rc = result.exit_code
            print(f"[{tag}] WARNING: myprint exit code {result.exit_code} (continuing)")
    return last_rc


def execute_print720(
    *,
    plan: Print720Plan,
    printer1: str,
    printer2: str,
    pdfs: List[PdfEntry],
    myprint_path: str,
    python_exe: Optional[str],
    inventory: Optional[ManualsInventory] = None,
    skip_collector: Optional[InventorySkipCollector] = None,
) -> None:
    has1 = len(plan.tasks_p1) > 0
    has2 = len(plan.tasks_p2) > 0

    if has1 and has2:
        print("\n[print720] Starting BOTH printers concurrently...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(
                    _run_tasks_for_printer,
                    printer=printer1,
                    tasks=plan.tasks_p1,
                    pdfs=pdfs,
                    myprint_path=myprint_path,
                    python_exe=python_exe,
                    tag="P1",
                    inventory=inventory,
                    skip_collector=skip_collector,
                ),
                ex.submit(
                    _run_tasks_for_printer,
                    printer=printer2,
                    tasks=plan.tasks_p2,
                    pdfs=pdfs,
                    myprint_path=myprint_path,
                    python_exe=python_exe,
                    tag="P2",
                    inventory=inventory,
                    skip_collector=skip_collector,
                ),
            ]
            for f in as_completed(futs):
                _ = f.result()
        print("\n[print720] Both printer queues completed.")
        return

    if has1:
        print("\n[print720] Only Printer 1 has work; running Printer 1 queue...")
        _run_tasks_for_printer(
            printer=printer1,
            tasks=plan.tasks_p1,
            pdfs=pdfs,
            myprint_path=myprint_path,
            python_exe=python_exe,
            tag="P1",
            inventory=inventory,
            skip_collector=skip_collector,
        )
        return

    if has2:
        print("\n[print720] Only Printer 2 has work; running Printer 2 queue...")
        _run_tasks_for_printer(
            printer=printer2,
            tasks=plan.tasks_p2,
            pdfs=pdfs,
            myprint_path=myprint_path,
            python_exe=python_exe,
            tag="P2",
            inventory=inventory,
            skip_collector=skip_collector,
        )
        return

    print("\n[print720] No eligible pages to print in this batch.")


# =============================================================================
# Reporting helpers
# =============================================================================


def print_inventory_skip_report(skip_collector: InventorySkipCollector) -> None:
    print("\n" + "=" * 78)
    print("MANUALS NOT PRINTED BECAUSE THEY ARE ALREADY IN INVENTORY")
    print("=" * 78)

    if not skip_collector.has_records():
        print("None.")
        return

    for rec in skip_collector.records():
        print(f"- {rec.pdf_base}")
        print(f"    PDF: {rec.pdf_path}")
        print(f"    Location: {rec.location}")


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--orders-csv", required=True, type=Path)
    ap.add_argument("--links-json", required=True, type=Path)
    ap.add_argument("--out-links-json", required=True, type=Path)

    ap.add_argument(
        "--pdf-folder",
        type=Path,
        default=Path(r"c:\Users\benoi\Downloads\ebay_manuals"),
        help="Folder containing PDFs (default: c:\\Users\\benoi\\Downloads\\ebay_manuals)",
    )
    ap.add_argument(
        "--pdf-folder2",
        type=Path,
        default=Path(r"c:\Users\benoi\Downloads\Manuals"),
        help="Optional second PDF folder (default: c:\\Users\\benoi\\Downloads\\Manuals)",
    )
    ap.add_argument("--recursive", action="store_true", help="Scan PDFs recursively under both folders")

    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument("--min-margin", type=float, default=8.0)

    ap.add_argument("--manuals-csv", type=Path, default=Path("manuals.csv"),
                    help="Printed-manual inventory CSV kept for compatibility (default: manuals.csv in current directory)")
    ap.add_argument("--inventory-csv", type=Path, default=DEFAULT_MANUALS_CSV,
                    help="manuals.csv used to detect manuals already in inventory before calling myprint")

    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="After selecting a PDF, run myprint.py using auto-inputs (no changes to myprint.py).")
    ap.add_argument("--myprint", default="myprint.py",
                    help="Path to myprint.py (default: myprint.py in current directory).")
    ap.add_argument("--python", default=None,
                    help="Python executable to run myprint.py (default: current interpreter).")

    ap.add_argument("--printer", type=str, default="", help="Default printer selection (e.g. 1).")
    ap.add_argument("--printer2", type=str, default="", help="Second printer selection (print720 mode).")
    ap.add_argument("--always-ask-printer", action="store_true",
                    help="Ask printer number for every print (normal mode).")

    ap.add_argument("--max-orders", type=int, default=0,
                    help="Optional limit for debugging (0 = no limit).")

    ap.add_argument("--print360", action="store_true",
                    help="Special mode: print up to 360 pages with no user intervention (except choosing printer).")
    ap.add_argument("--print720", action="store_true",
                    help="Special mode: dry-run split then print ~360 pages on printer1 and ~360 on printer2 concurrently.")
    ap.add_argument(
        "--print720-state",
        type=Path,
        default=Path("print720_state.json"),
        help="print720 persistent state JSON file (default: print720_state.json in current directory)",
    )
    ap.add_argument(
        "--print720-reset",
        action="store_true",
        help="Reset print720 progress (delete state file) and start from the beginning",
    )
    ap.add_argument(
        "--typewriter",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="For print720 only: select which printer is the 'typewriter' (old toner). "
             "0=none, 1=printer1, 2=printer2. The typewriter printer prints ONLY manuals "
             "with links-json flag {\"typewriter\": true} (optional; default false).",
    )

    args = ap.parse_args()

    orders = read_orders_csv(args.orders_csv)
    if args.max_orders:
        orders = orders[:args.max_orders]

    links = load_links_json(args.links_json)
    itemid_index, itemid_typewriter = build_itemid_index_and_flags(links)

    pdfs_1 = list_pdfs(args.pdf_folder, args.recursive)
    pdfs_2: List[PdfEntry] = []
    if args.pdf_folder2 and args.pdf_folder2.exists():
        pdfs_2 = list_pdfs(args.pdf_folder2, args.recursive)

    pdfs_by_norm: Dict[str, PdfEntry] = {}
    for p in pdfs_1 + pdfs_2:
        key = _norm(p.base)
        if key not in pdfs_by_norm:
            pdfs_by_norm[key] = p
    pdfs = list(pdfs_by_norm.values())

    if not pdfs:
        print(
            "No PDFs found in:\n"
            f"  - {args.pdf_folder}\n"
            f"  - {args.pdf_folder2}\n"
            f"(recursive={args.recursive})"
        )
        sys.exit(2)

    pdf_by_normbase: Dict[str, PdfEntry] = {_norm(p.base): p for p in pdfs}

    _ = load_manuals_from_csv_any(args.manuals_csv)
    inventory = ManualsInventory.load(args.inventory_csv)
    skip_collector = InventorySkipCollector()

    print(f"Loaded {len(orders)} orders from: {args.orders_csv}")
    print(f"Loaded {len(links)} existing link entries from: {args.links_json}")
    print(
        f"Indexed {len(pdfs)} unique PDF base names from:\n"
        f"  - {args.pdf_folder}\n"
        f"  - {args.pdf_folder2}\n"
        f"(recursive={args.recursive})"
    )
    print(f"Inventory CSV for skip detection: {inventory.csv_path}")

    default_printer = (args.printer or "").strip()

    try:
        if args.print720:
            printer1 = default_printer
            printer2 = (args.printer2 or "").strip()

            if not printer1:
                printer1 = input("\n[print720] Printer 1 number (e.g. 1): ").strip()
            if not printer2:
                printer2 = input("[print720] Printer 2 number (e.g. 2): ").strip()

            if not printer1:
                print("[print720] ERROR: printer1 is required.")
                sys.exit(2)
            if not printer2:
                print("[print720] ERROR: printer2 is required.")
                sys.exit(2)

            typewriter_printer = int(args.typewriter or 0)
            state_path: Path = args.print720_state

            if args.print720_reset:
                reset_print720_state(state_path)

            st = load_print720_state(state_path)
            posN = st.normal_pos()
            posT = st.typewriter_pos()

            print(
                f"\n[print720] Starting from state:"
                f"\n  NORMAL:     doc_list_index={posN.doc_list_index}, next_page={posN.next_page}"
                f"\n  TYPEWRITER: doc_list_index={posT.doc_list_index}, next_page={posT.next_page}"
            )
            print(f"[print720] State file: {state_path.resolve()}")

            while True:
                plan = plan_print720(
                    orders=orders,
                    itemid_index=itemid_index,
                    itemid_typewriter=itemid_typewriter,
                    pdf_by_normbase=pdf_by_normbase,
                    start_pos_normal=posN,
                    start_pos_typewriter=posT,
                    limit_each=360,
                    typewriter_printer=typewriter_printer,
                )

                _print_plan_summary(plan, typewriter_printer)
                total_pages = plan.printed_p1 + plan.printed_p2

                if total_pages <= 0:
                    print("\n[print720] Nothing eligible to print. Exiting.")
                    save_print720_state(state_path, Print720State.from_positions(posN, posT))
                    break

                execute_print720(
                    plan=plan,
                    printer1=printer1,
                    printer2=printer2,
                    pdfs=pdfs,
                    myprint_path=args.myprint,
                    python_exe=args.python,
                    inventory=inventory,
                    skip_collector=skip_collector,
                )

                posN = plan.end_pos_normal
                posT = plan.end_pos_typewriter
                save_print720_state(state_path, Print720State.from_positions(posN, posT))

                has_more = plan.has_more_normal or plan.has_more_typewriter
                if has_more:
                    ans = input("\n[print720] Do you want to continue printing the next batch? [y/N]: ").strip().lower()
                    if not ans.startswith("y"):
                        print("[print720] Stopping now; progress saved for next run.")
                        break
                else:
                    print("\n[print720] No more eligible pages after this batch. Done.")
                    try:
                        if state_path.exists():
                            state_path.unlink()
                    except Exception as e:
                        print(f"[print720] WARNING: failed to delete state file {state_path}: {e}")
                    break

            save_links_json(args.out_links_json, links)
            return

        if args.print360:
            if not default_printer:
                default_printer = input("\n[print360] Printer number (e.g. 1 or 2): ").strip()
            if not default_printer:
                print("[print360] ERROR: printer is required.")
                sys.exit(2)

            next_idx = 0
            resume: Optional[Print360Resume] = None
            batch_no = 1

            while True:
                next_idx, pages_printed, resume = run_print360_batch(
                    orders=orders,
                    start_index=next_idx,
                    start_resume=resume,
                    itemid_index=itemid_index,
                    pdf_by_normbase=pdf_by_normbase,
                    pdfs=pdfs,
                    printer=default_printer,
                    myprint_path=args.myprint,
                    python_exe=args.python,
                    page_limit=360,
                    inventory=inventory,
                    skip_collector=skip_collector,
                )

                print(f"\n[print360] Batch {batch_no} complete. Pages printed in this batch: {pages_printed}/360")

                has_more_orders = (resume is not None) or (next_idx < len(orders))
                if not has_more_orders:
                    print("\n[print360] No more eligible orders/pages after this batch. Done.")
                    save_links_json(args.out_links_json, links)
                    return

                if pages_printed <= 0:
                    print("\n[print360] No pages were printed in this batch. Stopping to avoid an infinite loop.")
                    save_links_json(args.out_links_json, links)
                    return

                ans = input(
                    "\n[print360] 360-page batch complete. Change paper if needed. "
                    "Continue with the next 360-page batch? [y/N]: "
                ).strip().lower()
                if not ans.startswith("y"):
                    print("[print360] Stopping now.")
                    if resume is not None:
                        print(
                            f"[print360] Next run should resume order index {resume.order_index}, "
                            f"manual '{resume.pdf.base}', page {resume.next_page}."
                        )
                    save_links_json(args.out_links_json, links)
                    return

                batch_no += 1

        else:
            start_index_for_normal = 0
            if args.do_print and not default_printer and not args.always_ask_printer:
                default_printer = input("\nDefault printer number for this run (e.g. 1 or 2): ").strip()

        updated = 0
        processed = 0

        for i in range(start_index_for_normal, len(orders)):
            row = orders[i]
            processed += 1

            title = (row.get("title") or "").strip()
            url = (row.get("item_url") or "").strip()
            item_id = (row.get("item_id") or "").strip()

            if not title or not url:
                continue

            chosen_pdf: Optional[PdfEntry] = None

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

            if chosen_pdf is None:
                cands = top_candidates(title, pdfs, k=3)
                chosen_pdf = choose_match_interactive(title, cands, args.min_score, args.min_margin)
                if not chosen_pdf:
                    print("No match selected. Moving on.")
                    continue

            links.setdefault(chosen_pdf.base, {})
            links[chosen_pdf.base]["url"] = url
            if item_id:
                links[chosen_pdf.base]["item_id"] = item_id
                itemid_index[item_id] = chosen_pdf.base
                rec = links.get(chosen_pdf.base, {})
                itemid_typewriter[item_id] = _as_bool(rec.get("typewriter", False))

            updated += 1
            print(f"Linked: {chosen_pdf.base}  ->  {url}   (item_id={item_id})")

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

                result = myprint_auto_print_range(
                    pdfs=pdfs,
                    chosen_pdf=chosen_pdf,
                    printer=prn,
                    page_range=page_range,
                    myprint_path=args.myprint,
                    python_exe=args.python,
                    inventory=inventory,
                    skip_collector=skip_collector,
                )
                if result.skipped_in_inventory:
                    print("Manual already exists in inventory. Not printed.")
                elif result.exit_code != 0:
                    print(f"WARNING: myprint.py returned exit code {result.exit_code}. Continuing.")

        save_links_json(args.out_links_json, links)
        print(f"\nDone. Updated/added {updated} links. Processed {processed} order rows.")
    finally:
        print_inventory_skip_report(skip_collector)


if __name__ == "__main__":
    main()
