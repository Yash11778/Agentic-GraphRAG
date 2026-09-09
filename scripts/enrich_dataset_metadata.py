"""
Add structural metadata (case name, court, year) to the Pile-of-Law-derived
legal corpus by parsing each opinion's own citation header using eyecite --
the actual citation-parsing library Free Law Project (CourtListener's
creator) uses internally. This resolves reporter abbreviations against a
real reporters database (e.g. "Ga. App." -> Georgia Appeals Reports, state
court), avoiding the mismatches a hand-rolled substring matcher hits (an
earlier version of this script confused "Ga." (Supreme Court) with "Ga. App."
(Court of Appeals) because CourtListener's own citation_string field for the
Court of Appeals is "Ga. Ct. App.", not the "Ga. App." actually used in
citations -- eyecite's reporters-db doesn't have this mismatch).

Case name is taken from eyecite's parsed plaintiff/defendant metadata when
available (works well for vendor-neutral citations like "NY Slip Op"), else
falls back to a single-line party-caption regex ("X v. Y") scanned from the
opinion's own header text.

Usage:
    python scripts/enrich_dataset_metadata.py
"""
import json
import re
from pathlib import Path

from eyecite import get_citations
from eyecite.models import FullCaseCitation
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.resolve()
IN_FILE = ROOT / "data" / "raw" / "dataset_100m.jsonl"
OUT_FILE = ROOT / "data" / "raw" / "dataset_100m_enriched.jsonl"

HEADER_WINDOW = 600

# Single physical line only (no \s, which includes \n) -- an earlier version
# using \s let the match greedily swallow the whole multi-line header block.
CASE_NAME_RE = re.compile(
    r"^[A-Z][A-Za-z0-9.,'&\-\s]{2,80}?\s+v(?:s)?\.?\s+[A-Za-z0-9.,'&\-]{2,80}\.?$",
    re.MULTILINE,
)


_V_LINE_RE = re.compile(r"^v\.?s?\.?$", re.IGNORECASE)

# Slip-opinion / order headers with no reporter citation at all, e.g.
# "IN THE SUPREME COURT OF PENNSYLVANIA" or "COURT OF APPEALS OF GEORGIA" --
# eyecite finds nothing here since there's no citation, but the court name
# is stated in plain text right at the top.
_COURT_HEADER_RE = re.compile(
    r"\b((?:SUPREME|SUPERIOR|CIRCUIT|DISTRICT|APPELLATE|CHANCERY)\s+COURT"
    r"(?:\s+OF\s+(?:APPEALS?|CRIMINAL\s+APPEALS?|CHANCERY))?"
    r"\s+OF\s+[A-Z][A-Za-z .]{2,40}"
    r"|COURT\s+OF\s+(?:APPEALS?|CRIMINAL\s+APPEALS?|CHANCERY)\s+OF\s+[A-Z][A-Za-z .]{2,40})",
    re.IGNORECASE,
)


def find_court_header_fallback(text: str) -> str | None:
    m = _COURT_HEADER_RE.search(text[:HEADER_WINDOW])
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip().title()
    return None


def find_case_name_fallback(text: str) -> str | None:
    lines = [l.strip() for l in text[:HEADER_WINDOW].splitlines()]

    # Single-line caption: "ROBINSON v. THE STATE"
    for line in lines:
        m = CASE_NAME_RE.match(line)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()

    # Multi-line caption: "ROBINSON" / "v." / "THE STATE." (common in older reporters)
    for i, line in enumerate(lines):
        if _V_LINE_RE.match(line):
            before = next((lines[j] for j in range(i - 1, -1, -1) if lines[j]), None)
            after = next((lines[j] for j in range(i + 1, len(lines)) if lines[j]), None)
            if before and after and len(before) < 100 and len(after) < 100:
                return f"{before.rstrip('.')} v. {after.rstrip('.')}"
    return None


def extract_metadata(text: str) -> dict:
    header = text[:HEADER_WINDOW]
    cites = [c for c in get_citations(header) if isinstance(c, FullCaseCitation)]

    court_id = court_name = year = case_name = None
    if cites:
        c = cites[0]
        year = c.year
        plaintiff = c.metadata.plaintiff
        defendant = c.metadata.defendant
        if plaintiff and defendant:
            case_name = f"{plaintiff} v. {defendant}"
        edition = c.edition_guess
        if edition and edition.reporter:
            court_id = edition.reporter.short_name
            court_name = edition.reporter.name
            court_type = edition.reporter.cite_type  # 'state', 'federal', 'scotus', ...
        else:
            court_type = None
    else:
        court_type = None

    if not court_name:
        header_court = find_court_header_fallback(text)
        if header_court:
            court_name = header_court
            court_type = court_type or "state_or_local"  # header pattern can't distinguish reliably

    if not case_name:
        case_name = find_case_name_fallback(text)

    return {
        "case_name":  case_name,
        "court_id":   court_id,
        "court_name": court_name,
        "court_type": court_type,
        "year":       year,
    }


def main():
    total = 0
    matched_court = 0       # court_name OR court_id present (any confident signal)
    matched_case_name = 0   # real parsed case name, NOT the generic title fallback
    matched_year = 0

    with open(IN_FILE, encoding="utf-8") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Enriching"):
            row = json.loads(line)
            meta = extract_metadata(row["text"])

            # case_name stays null when not confidently parsed -- title (always
            # present, but often just the first line/citation) is kept as a
            # SEPARATE field so downstream consumers never mistake a raw text
            # fragment for a real party-caption case name.
            record = {
                "id":         row["id"],
                "title":      row.get("title"),
                "case_name":  meta["case_name"],
                "court_id":   meta["court_id"],
                "court_name": meta["court_name"],
                "court_type": meta["court_type"],
                "year":       meta["year"],
                "text":       row["text"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            matched_court += bool(meta["court_id"] or meta["court_name"])
            matched_case_name += bool(meta["case_name"])
            matched_year += bool(meta["year"])

    print(f"\nTotal opinions          : {total:,}")
    print(f"Court identified        : {matched_court:,} ({matched_court/total*100:.1f}%)")
    print(f"Real case name parsed   : {matched_case_name:,} ({matched_case_name/total*100:.1f}%)")
    print(f"Year identified         : {matched_year:,} ({matched_year/total*100:.1f}%)")
    print(f"(Remaining records keep 'title' as a generic label; entity extraction")
    print(f" in the graph pipeline will still recover names/courts from full body text)")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
