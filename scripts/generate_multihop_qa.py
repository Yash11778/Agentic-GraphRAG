"""
Generate a multi-hop QA set grounded in REAL citation chains from citations.csv,
not random single-passage sampling. Each question is built from an actual chain
of cases that cite each other in the corpus (A cites B [cites C]), so answering
it correctly requires connecting facts across multiple documents -- exactly the
structure Basic RAG (flat chunk similarity) cannot reliably traverse, and
GraphRAG's citation-graph retrieval (pipelines/pipeline3_graphrag.py) is built
to handle.

Mix: mostly 2-hop chains (verified plentiful, ~3,580 in the full corpus) with
some 3-hop attempts (verified to exist, rarer) per user's chosen QA difficulty
mix. Each chain's actual case texts are given to Gemini to write one genuine
multi-hop question + ground-truth answer -- not a single-document fact.

Usage:
    python scripts/generate_multihop_qa.py --target-2hop 20 --target-3hop 8
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import setup_gemini, gemini_generate

ROOT = Path(__file__).parent.parent.resolve()
CITATIONS_FILE = ROOT / "data" / "raw" / "citations.csv"
DATASET_FILE = ROOT / "data" / "raw" / "dataset_100m_enriched.jsonl"
OUT_FILE = ROOT / "data" / "qa" / "qa_pairs_legal_multihop.json"

PROMPT_2HOP = """You are given two real court opinions where the SECOND case cites the FIRST case.
Write ONE genuine multi-hop question that can ONLY be answered by using facts from BOTH opinions together \
(not answerable from either one alone), plus a precise, complete-sentence ground-truth answer.

Good multi-hop question patterns: "What principle did [case B] rely on from [case A], and how did it apply that \
principle to a new context?", "How did the court in [case B] extend or distinguish the holding of [case A]?"

IMPORTANT:
- Refer to each case by its real case name throughout the question AND answer -- never write "Case A" or "Case B".
- In the QUESTION, the FIRST time you name each case, append its citation id in parentheses exactly like this: \
Case Name ({case_a_id}) for Case A and Case Name ({case_b_id}) for Case B. This is how a real legal query pins \
down which specific opinion is meant when several share a caption. Do NOT put the id in the answer, only the question.

Case A (cited, earlier): {case_a_name}
{case_a_text}

Case B (citing, later): {case_b_name}
{case_b_text}

Output ONLY valid JSON, no markdown:
{{"question": "...", "answer": "...", "hop_count": 2, "case_ids": ["{case_a_id}", "{case_b_id}"]}}"""

PROMPT_3HOP = """You are given three real court opinions forming a citation chain: Case C cites Case B, and Case B cites Case A.
Write ONE genuine multi-hop question that requires tracing the chain across ALL THREE opinions \
(not answerable from any one or two alone), plus a precise, complete-sentence ground-truth answer.

Good pattern: "What principle originated in [case A], how did [case B] apply it, and how did [case C] in turn \
build on or distinguish [case B]'s application of it?"

IMPORTANT:
- Refer to each case by its real case name throughout the question AND answer -- \
never write "Case A", "Case B", or "Case C" literally.
- In the QUESTION, the FIRST time you name each case, append its citation id in parentheses \
exactly like this: Case Name ({case_a_id}) for Case A, Case Name ({case_b_id}) for Case B, \
Case Name ({case_c_id}) for Case C. This is how a real legal query pins down which specific \
opinion is meant when several share a caption. Do NOT put the id in the answer, only the question.

Case A (earliest, cited by B): {case_a_name}
{case_a_text}

Case B (cites A, cited by C): {case_b_name}
{case_b_text}

Case C (citing, latest): {case_c_name}
{case_c_text}

Output ONLY valid JSON, no markdown:
{{"question": "...", "answer": "...", "hop_count": 3, "case_ids": ["{case_a_id}", "{case_b_id}", "{case_c_id}"]}}"""

TEXT_SNIPPET = 2500  # chars per case fed to the generator prompt


def load_citation_graph():
    citing_to_cited = defaultdict(set)
    with open(CITATIONS_FILE, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for citing, cited in r:
            citing_to_cited[citing].add(cited)
    return citing_to_cited


def find_2hop_chains(citing_to_cited: dict, n: int, seed: int) -> list[tuple]:
    chains = []
    for a, bs in citing_to_cited.items():
        for b in bs:
            chains.append((b, a))  # (citing, cited) = (later case, earlier case)
    random.Random(seed).shuffle(chains)
    return chains[:n]


def find_3hop_chains(citing_to_cited: dict, n: int, seed: int) -> list[tuple]:
    chains = []
    items = list(citing_to_cited.items())
    random.Random(seed).shuffle(items)
    for a, bs in items:
        for b in bs:
            for c in citing_to_cited.get(b, set()):
                if c != a:
                    chains.append((a, b, c))  # (citing, cited, cited-by-cited)
                    break
            if len(chains) >= n:
                break
        if len(chains) >= n:
            break
    return chains[:n]


def load_case(cache: dict, case_id: str) -> dict | None:
    return cache.get(case_id)


def build_case_cache(all_ids: set) -> dict:
    cache = {}
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in all_ids:
                cache[row["id"]] = row
    return cache


def gen_2hop(client, cache, citing_id, cited_id) -> dict | None:
    ca, cb = load_case(cache, cited_id), load_case(cache, citing_id)
    if not ca or not cb:
        return None
    prompt = PROMPT_2HOP.format(
        case_a_name=ca.get("case_name") or ca.get("title") or cited_id,
        case_a_text=ca["text"][:TEXT_SNIPPET],
        case_b_name=cb.get("case_name") or cb.get("title") or citing_id,
        case_b_text=cb["text"][:TEXT_SNIPPET],
        case_a_id=cited_id, case_b_id=citing_id,
    )
    raw = gemini_generate(client, prompt, max_tokens=400)
    qa = _parse_json(raw)
    if qa is not None:
        qa["hop_count"] = 2  # stamp defensively -- the model occasionally omits it
    return qa


def gen_3hop(client, cache, citing_id, mid_id, cited_id) -> dict | None:
    ca, cb, cc = load_case(cache, cited_id), load_case(cache, mid_id), load_case(cache, citing_id)
    if not ca or not cb or not cc:
        return None
    prompt = PROMPT_3HOP.format(
        case_a_name=ca.get("case_name") or ca.get("title") or cited_id,
        case_a_text=ca["text"][:TEXT_SNIPPET],
        case_b_name=cb.get("case_name") or cb.get("title") or mid_id,
        case_b_text=cb["text"][:TEXT_SNIPPET],
        case_c_name=cc.get("case_name") or cc.get("title") or citing_id,
        case_c_text=cc["text"][:TEXT_SNIPPET],
        case_a_id=cited_id, case_b_id=mid_id, case_c_id=citing_id,
    )
    raw = gemini_generate(client, prompt, max_tokens=500)
    qa = _parse_json(raw)
    if qa is not None:
        qa["hop_count"] = 3  # stamp defensively -- the model occasionally omits it
    return qa


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return None


def main(target_2hop: int, target_3hop: int, seed: int):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = setup_gemini()

    print("Loading citation graph...")
    citing_to_cited = load_citation_graph()

    print(f"Sampling {target_2hop} 2-hop and {target_3hop} 3-hop candidate chains...")
    two_hop = find_2hop_chains(citing_to_cited, target_2hop * 2, seed)  # oversample, some will fail to parse
    three_hop = find_3hop_chains(citing_to_cited, target_3hop * 2, seed)

    all_ids = set()
    for citing, cited in two_hop:
        all_ids.add(citing); all_ids.add(cited)
    for citing, mid, cited in three_hop:
        all_ids.add(citing); all_ids.add(mid); all_ids.add(cited)

    print(f"Loading {len(all_ids):,} case texts...")
    cache = build_case_cache(all_ids)

    # Resume from whatever a previous (possibly crashed) run already saved, rather
    # than starting over and re-spending quota/API calls on chains already done.
    results = []
    if OUT_FILE.exists():
        try:
            results = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            print(f"Resuming: {len(results)} questions already saved from a previous run")
        except Exception:
            results = []
    done_case_sets = {tuple(sorted(r.get("case_ids", []))) for r in results}

    def save():
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Generating 2-hop questions ---")
    for citing, cited in two_hop:
        if len([r for r in results if r.get("hop_count") == 2]) >= target_2hop:
            break
        if tuple(sorted([citing, cited])) in done_case_sets:
            continue
        try:
            qa = gen_2hop(client, cache, citing, cited)
        except Exception as e:
            print(f"  [2-hop] SKIP (error: {e})")
            continue
        if qa and qa.get("question") and qa.get("answer"):
            results.append(qa)
            save()  # incremental -- a later crash never loses earlier progress
            print(f"  [2-hop] {qa['question'][:80]}")

    print("\n--- Generating 3-hop questions ---")
    for citing, mid, cited in three_hop:
        if len([r for r in results if r.get("hop_count") == 3]) >= target_3hop:
            break
        if tuple(sorted([citing, mid, cited])) in done_case_sets:
            continue
        try:
            qa = gen_3hop(client, cache, citing, mid, cited)
        except Exception as e:
            print(f"  [3-hop] SKIP (error: {e})")
            continue
        if qa and qa.get("question") and qa.get("answer"):
            results.append(qa)
            save()
            print(f"  [3-hop] {qa['question'][:80]}")

    save()
    n2 = len([r for r in results if r.get("hop_count") == 2])
    n3 = len([r for r in results if r.get("hop_count") == 3])
    print(f"\nDone! {len(results)} questions written ({n2} 2-hop, {n3} 3-hop)")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-2hop", type=int, default=20)
    parser.add_argument("--target-3hop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.target_2hop, args.target_3hop, args.seed)
