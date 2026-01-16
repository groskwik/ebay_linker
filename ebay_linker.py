##!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------
# Normalization / similarity
# ----------------------------

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
    """
    Return number of pages in PDF (int) or None if unreadable.
    Uses pypdf if available.
    """
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
# Printed manual inventory CSV (optional) - kept as-is
# ----------------------------

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


# ----------------------------
# Matching helpers (orders -> PDFs) (normal mode)
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

def run_myprint_with_auto_inputs(myprint_path: str, python_exe: Optional[str], auto_inputs: List[str]) -> int:
    py = python_exe or sys.executable
    cmd = [py, myprint_path]
    payload = "\n".join(auto_inputs) + "\n"

    print("\n=== Running myprint.py with auto inputs ===")
    print("Command:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
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
) -> int:
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
    return run_myprint_with_auto_inputs(myprint_path, python_exe, auto_inputs)


# ----------------------------
# print360 mode (unchanged, kept for compatibility)
# ----------------------------

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
    itemid_index: Dict[str, str],
    pdf_by_normbase: Dict[str, PdfEntry],
    pdfs: List[PdfEntry],
    printer: str,
    myprint_path: str,
    python_exe: Optional[str],
    page_limit: int = 360,
) -> Tuple[int, int, Optional[Print360Resume]]:
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
            continue

        if not item_id or item_id not in itemid_index:
            print(f"\n[print360] SKIP (not in links DB): item_id={item_id!r}  title={title}")
            idx += 1
            continue

        pdf_base = itemid_index[item_id]
        pdf = pdf_by_normbase.get(_norm(pdf_base))
        if not pdf:
            print(f"\n[print360] SKIP (PDF not found): item_id={item_id}  pdf_base={pdf_base!r}")
            idx += 1
            continue

        total_pages = get_pdf_pagecount(pdf.path)
        if not total_pages or total_pages <= 0:
            print(f"\n[print360] SKIP (cannot read page count): {pdf.path}")
            idx += 1
            continue

        remaining_capacity = page_limit - pages_printed
        if remaining_capacity <= 0:
            break

        if total_pages <= remaining_capacity:
            pr = f"1-{total_pages}"
            print(f"\n[print360] PRINT FULL: {pdf.base}  pages={pr}  (total={total_pages})")
            rc = myprint_auto_print_range(
                pdfs=pdfs, chosen_pdf=pdf, printer=printer, page_range=pr,
                myprint_path=myprint_path, python_exe=python_exe,
            )
            if rc != 0:
                print(f"[print360] WARNING: myprint exit code {rc} (continuing)")
            pages_printed += total_pages
            idx += 1
            continue

        start_page = 1
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
        rc = myprint_auto_print_range(
            pdfs=pdfs, chosen_pdf=pdf, printer=printer, page_range=pr,
            myprint_path=myprint_path, python_exe=python_exe,
        )
        if rc != 0:
            print(f"[print360] WARNING: myprint exit code {rc} (continuing)")

        pages_printed += printed_now
        if end_page < total_pages:
            resume = Print360Resume(order_index=idx, pdf=pdf, total_pages=total_pages, next_page=end_page + 1)
        break

    next_start_index = idx
    return next_start_index, pages_printed, resume

def finish_resume_manual(
    *,
    resume: Print360Resume,
    printer: str,
    pdfs: List[PdfEntry],
    myprint_path: str,
    python_exe: Optional[str],
) -> None:
    if resume.next_page > resume.total_pages:
        return
    pr = f"{resume.next_page}-{resume.total_pages}"
    print(f"\n[resume] FINISH MANUAL: {resume.pdf.base}  pages={pr}  (total={resume.total_pages})")
    rc = myprint_auto_print_range(
        pdfs=pdfs, chosen_pdf=resume.pdf, printer=printer, page_range=pr,
        myprint_path=myprint_path, python_exe=python_exe,
    )
    if rc != 0:
        print(f"[resume] WARNING: myprint exit code {rc} (continuing)")


# ----------------------------
# print720 mode (NEW)
# ----------------------------

@dataclass
class EligibleDoc:
    order_index: int     # index in the original orders list
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
    doc_list_index: int  # index into eligible_docs list
    next_page: int       # next page within that doc (1-based)

@dataclass
class Print720Plan:
    tasks_p1: List[PrintTask]
    tasks_p2: List[PrintTask]
    printed_p1: int
    printed_p2: int
    end_pos: PrintStreamPos
    has_more: bool

def _build_eligible_docs(
    *,
    orders: List[Dict[str, str]],
    itemid_index: Dict[str, str],
    pdf_by_normbase: Dict[str, PdfEntry],
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
    # If next_page is past total, move to next doc
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
) -> Tuple[List[PrintTask], int, PrintStreamPos, bool]:
    """
    Plans print tasks for one printer by consuming up to page_limit pages from the stream.
    If the allocation ends mid-document, it forces end_page to an even number (may add 1 page).
    Returns: (tasks, pages_allocated, end_pos, ended_on_cut)
    """
    tasks: List[PrintTask] = []
    allocated = 0
    pos = _advance_pos(docs, start_pos)
    ended_on_cut = False

    while pos.doc_list_index < len(docs) and allocated < page_limit:
        d = docs[pos.doc_list_index]
        start_page = pos.next_page
        remaining_in_doc = d.total_pages - start_page + 1
        remaining_capacity = page_limit - allocated

        if remaining_in_doc <= remaining_capacity:
            # Take the rest of the doc
            end_page = d.total_pages
            tasks.append(PrintTask(pdf=d.pdf, start_page=start_page, end_page=end_page))
            allocated += (end_page - start_page + 1)
            pos = PrintStreamPos(doc_list_index=pos.doc_list_index + 1, next_page=1)
            pos = _advance_pos(docs, pos)
            continue

        # Need to cut inside the doc
        end_page = start_page + remaining_capacity - 1

        if force_even_end_if_cut and (end_page % 2 == 1):
            end_page += 1  # may exceed limit by 1, but preserves duplex parity

        if end_page > d.total_pages:
            end_page = d.total_pages
            # if clamped to odd total, step down to even when possible
            if force_even_end_if_cut and (end_page % 2 == 1) and (end_page - 1 >= start_page):
                end_page -= 1

        tasks.append(PrintTask(pdf=d.pdf, start_page=start_page, end_page=end_page))
        allocated += max(0, end_page - start_page + 1)
        pos = PrintStreamPos(doc_list_index=pos.doc_list_index, next_page=end_page + 1)
        pos = _advance_pos(docs, pos)
        ended_on_cut = True
        break

    return tasks, allocated, pos, ended_on_cut

def plan_print720(
    *,
    orders: List[Dict[str, str]],
    itemid_index: Dict[str, str],
    pdf_by_normbase: Dict[str, PdfEntry],
    start_pos: PrintStreamPos,
    limit_each: int = 360,
) -> Print720Plan:
    """
    Dry-run planner:
      - Builds eligible docs (item must be in links DB, PDF exists, page count readable)
      - Consumes the stream:
          first limit_each pages -> printer 1
          next  limit_each pages -> printer 2
      - Cuts inside a doc end on even page (duplex parity) when needed.
    """
    docs = _build_eligible_docs(orders=orders, itemid_index=itemid_index, pdf_by_normbase=pdf_by_normbase)

    # start_pos is in the "eligible docs stream" coordinates
    pos0 = _advance_pos(docs, start_pos)

    t1, p1, pos1, _ = _plan_for_one_printer(docs=docs, start_pos=pos0, page_limit=limit_each, force_even_end_if_cut=True)
    t2, p2, pos2, _ = _plan_for_one_printer(docs=docs, start_pos=pos1, page_limit=limit_each, force_even_end_if_cut=True)

    # Do we have more after this batch?
    pos2a = _advance_pos(docs, pos2)
    has_more = pos2a.doc_list_index < len(docs)

    return Print720Plan(
        tasks_p1=t1,
        tasks_p2=t2,
        printed_p1=p1,
        printed_p2=p2,
        end_pos=pos2a,
        has_more=has_more,
    )

def _print_plan_summary(plan: Print720Plan) -> None:
    print("\n[print720] Dry run plan:")
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
) -> int:
    """
    Runs a list of tasks sequentially for one printer.
    Returns last non-zero rc if any; else 0.
    """
    last_rc = 0
    for t in tasks:
        print(f"\n[{tag}] PRINT: {t.pdf.base}  pages={t.page_range}  on printer={printer}")
        rc = myprint_auto_print_range(
            pdfs=pdfs,
            chosen_pdf=t.pdf,
            printer=printer,
            page_range=t.page_range,
            myprint_path=myprint_path,
            python_exe=python_exe,
        )
        if rc != 0:
            last_rc = rc
            print(f"[{tag}] WARNING: myprint exit code {rc} (continuing)")
    return last_rc

def execute_print720(
    *,
    plan: Print720Plan,
    printer1: str,
    printer2: str,
    pdfs: List[PdfEntry],
    myprint_path: str,
    python_exe: Optional[str],
) -> None:
    """
    Executes printer 1 and printer 2 queues concurrently (when both have tasks).
    If only one queue has tasks, runs only that one.
    """
    has1 = len(plan.tasks_p1) > 0
    has2 = len(plan.tasks_p2) > 0

    if has1 and has2:
        print("\n[print720] Starting BOTH printers concurrently...")
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = []
            futs.append(ex.submit(
                _run_tasks_for_printer,
                printer=printer1, tasks=plan.tasks_p1, pdfs=pdfs,
                myprint_path=myprint_path, python_exe=python_exe, tag="P1"
            ))
            futs.append(ex.submit(
                _run_tasks_for_printer,
                printer=printer2, tasks=plan.tasks_p2, pdfs=pdfs,
                myprint_path=myprint_path, python_exe=python_exe, tag="P2"
            ))
            for f in as_completed(futs):
                _ = f.result()
        print("\n[print720] Both printer queues completed.")
        return

    if has1:
        print("\n[print720] Only Printer 1 has work; running Printer 1 queue...")
        _run_tasks_for_printer(
            printer=printer1, tasks=plan.tasks_p1, pdfs=pdfs,
            myprint_path=myprint_path, python_exe=python_exe, tag="P1"
        )
        return

    if has2:
        print("\n[print720] Only Printer 2 has work; running Printer 2 queue...")
        _run_tasks_for_printer(
            printer=printer2, tasks=plan.tasks_p2, pdfs=pdfs,
            myprint_path=myprint_path, python_exe=python_exe, tag="P2"
        )
        return

    print("\n[print720] No eligible pages to print in this batch.")


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
    ap.add_argument("--pdf-folder2", type=Path, default=Path(r"c:\Users\benoi\Downloads\Manuals"),
                    help="Optional second PDF folder (default: c:\\Users\\benoi\\Downloads\\Manuals)")
    ap.add_argument("--recursive", action="store_true", help="Scan PDFs recursively under --pdf-folder")

    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument("--min-margin", type=float, default=8.0)

    ap.add_argument("--manuals-csv", type=Path, default=Path("manuals.csv"),
                    help="Printed-manual inventory CSV (default: manuals.csv in current directory)")

    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="After selecting a PDF, run myprint.py using auto-inputs (no changes to myprint.py).")
    ap.add_argument("--myprint", default="myprint.py",
                    help="Path to myprint.py (default: myprint.py in current directory).")
    ap.add_argument("--python", default=None,
                    help="Python executable to run myprint.py (default: current interpreter).")

    ap.add_argument("--printer", type=str, default="", help="Default printer selection (e.g. 1 or 2).")
    ap.add_argument("--printer2", type=str, default="", help="Second printer selection (print720 mode).")
    ap.add_argument("--always-ask-printer", action="store_true",
                    help="Ask printer number for every print (normal mode).")

    ap.add_argument("--max-orders", type=int, default=0,
                    help="Optional limit for debugging (0 = no limit).")

    ap.add_argument("--print360", action="store_true",
                    help="Special mode: print up to 360 pages with no user intervention (except choosing printer).")
    ap.add_argument("--print720", action="store_true",
                    help="Special mode: dry-run split then print ~360 pages on printer1 and ~360 on printer2 concurrently.")

    args = ap.parse_args()

    orders = read_orders_csv(args.orders_csv)
    if args.max_orders:
        orders = orders[:args.max_orders]

    links = load_links_json(args.links_json)
    itemid_index = build_itemid_index(links)

    # scan PDFs from two folders, dedupe by normalized base name (folder1 wins ties)
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

    manuals_map = load_manuals_from_csv_any(args.manuals_csv)
    _ = manuals_map  # kept for your other features (not required here)

    print(f"Loaded {len(orders)} orders from: {args.orders_csv}")
    print(f"Loaded {len(links)} existing link entries from: {args.links_json}")
    print(
        f"Indexed {len(pdfs)} unique PDF base names from:\n"
        f"  - {args.pdf_folder}\n"
        f"  - {args.pdf_folder2}\n"
        f"(recursive={args.recursive})"
    )

    default_printer = (args.printer or "").strip()

    # ----------------------------
    # print720 mode
    # ----------------------------
    if args.print720:
        # only user intervention allowed in this mode: selecting printers
        printer1 = default_printer
        printer2 = (args.printer2 or "").strip()

        if not printer1:
            printer1 = input("\n[print720] Printer 1 number (e.g. 1): ").strip()
        if not printer2:
            printer2 = input("[print720] Printer 2 number (e.g. 2): ").strip()

        if not printer1:
            print("[print720] ERROR: printer1 is required.")
            sys.exit(2)
        # printer2 may be empty in theory, but mode concept expects it; keep strict:
        if not printer2:
            print("[print720] ERROR: printer2 is required.")
            sys.exit(2)

        # Stream position is in ELIGIBLE DOC LIST coordinates.
        # Start at the beginning each run; within a run we support "continue printing" in a loop.
        stream_pos = PrintStreamPos(doc_list_index=0, next_page=1)

        while True:
            plan = plan_print720(
                orders=orders,
                itemid_index=itemid_index,
                pdf_by_normbase=pdf_by_normbase,
                start_pos=stream_pos,
                limit_each=360,
            )

            _print_plan_summary(plan)
            total_pages = plan.printed_p1 + plan.printed_p2

            if total_pages <= 0:
                print("\n[print720] Nothing eligible to print. Exiting.")
                break

            # execute concurrently based on the plan
            execute_print720(
                plan=plan,
                printer1=printer1,
                printer2=printer2,
                pdfs=pdfs,
                myprint_path=args.myprint,
                python_exe=args.python,
            )

            # advance stream position for potential next batch
            stream_pos = plan.end_pos

            # Ask to continue only if there is more beyond this ~720 batch
            if plan.has_more:
                ans = input("\n[print720] Do you want to continue printing the next batch? [y/N]: ").strip().lower()
                if not ans.startswith("y"):
                    break
            else:
                print("\n[print720] No more eligible pages after this batch. Done.")
                break

        save_links_json(args.out_links_json, links)
        return

    # ----------------------------
    # print360 mode (existing behavior)
    # ----------------------------
    if args.print360:
        args.do_print = True
        args.always_ask_printer = False

        if not default_printer:
            default_printer = input("\n[print360] Printer number (e.g. 1 or 2): ").strip()
        if not default_printer:
            print("[print360] ERROR: printer is required.")
            sys.exit(2)

        next_idx, pages_printed, resume = run_print360_batch(
            orders=orders,
            start_index=0,
            itemid_index=itemid_index,
            pdf_by_normbase=pdf_by_normbase,
            pdfs=pdfs,
            printer=default_printer,
            myprint_path=args.myprint,
            python_exe=args.python,
            page_limit=360,
        )

        print(f"\n[print360] Batch complete. Pages printed in this batch: {pages_printed}/360")

        if pages_printed >= 360:
            ans = input("\n[print360] Do you want to continue later? [y/N]: ").strip().lower()
            if not ans.startswith("y"):
                save_links_json(args.out_links_json, links)
                print("[print360] Stopping (no continue).")
                return

            if resume is not None:
                finish_resume_manual(
                    resume=resume,
                    printer=default_printer,
                    pdfs=pdfs,
                    myprint_path=args.myprint,
                    python_exe=args.python,
                )
                next_idx = resume.order_index + 1

            print("\n[print360] Continuing in NORMAL MODE from remaining orders...\n")
        else:
            print("\n[print360] Did not reach 360 pages. Continuing in NORMAL MODE...\n")

        start_index_for_normal = next_idx
    else:
        start_index_for_normal = 0
        if args.do_print and not default_printer and not args.always_ask_printer:
            default_printer = input("\nDefault printer number for this run (e.g. 1 or 2): ").strip()

    # ----------------------------
    # NORMAL MODE loop
    # ----------------------------
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

            rc = myprint_auto_print_range(
                pdfs=pdfs,
                chosen_pdf=chosen_pdf,
                printer=prn,
                page_range=page_range,
                myprint_path=args.myprint,
                python_exe=args.python,
            )
            if rc != 0:
                print(f"WARNING: myprint.py returned exit code {rc}. Continuing.")

    save_links_json(args.out_links_json, links)
    print(f"\nDone. Updated/added {updated} links. Processed {processed} order rows.")


if __name__ == "__main__":
    main()
