"""Audit response-letter coverage against the merged reviewer comments.

Builds the comment ledger used to drive the remaining revision work: parses the
19 reviewer comments from the merged comment file, checks which are addressed in
the response letter, and flags any response section still written as a promise
("we will ...") rather than as completed work.

Run from the project root:
    python scripts/audit_response_coverage.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMENTS = ROOT / "Animals_Round1_Reviewers_Comments_merged.txt"
RESPONSE = ROOT / "Animals_Round1_Response_EN.md"
OUT = ROOT / "results" / "animals_round1"

PROMISE_MARKERS = (
    "we will ", "the revised version will", "will be added", "will be supplemented",
    "to be added", "will be reported", "will be completed", "in progress",
    "will re-examine", "will document", "will report", "will use", "will also",
)


def parse_comments() -> list[dict]:
    text = COMMENTS.read_text(encoding="utf-8")
    r1_block, rest = text.split("Reviewer 2\n", 1)
    r2_body = rest.split("Comments and Suggestions for Authors\n", 1)[1]

    comments: list[dict] = []
    for para in r1_block.split("\n\n"):
        para = para.strip()
        m = re.match(r"^([1-6])\.\s+(.*)", para, flags=re.S)
        if m:
            comments.append({"id": f"R1-{m.group(1)}", "reviewer": 1,
                             "n": int(m.group(1)), "text": m.group(2).strip()})

    paras = [p.strip() for p in r2_body.split("\n\n") if p.strip()]
    body = paras[1:14]
    for i, para in enumerate(body, start=1):
        comments.append({"id": f"R2-{i}", "reviewer": 2, "n": i, "text": para.strip()})
    return comments


def split_response_sections(response: str) -> tuple[list[str], list[str]]:
    """Split the response into the 6 Reviewer 1 and 13 Reviewer 2 bodies, in order.

    Both reviewers number their comments from 1, so R1 and R2 headings repeat the
    same numbers. Sections are therefore split at "## Reviewer N" first and only
    then at "### Comment N" within each reviewer block.
    """
    blocks: dict[int, str] = {}
    headings = list(re.finditer(r"^##\s*Reviewer\s+(\d)\s*$", response, flags=re.M))
    for i, m in enumerate(headings):
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(response)
        blocks[int(m.group(1))] = response[start:end]

    def bodies(block: str) -> list[str]:
        marks = list(re.finditer(r"^###\s*Comment\s+(\d+)", block, flags=re.M))
        out = []
        for i, m in enumerate(marks):
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(block)
            out.append(block[start:end])
        return out

    return bodies(blocks.get(1, "")), bodies(blocks.get(2, ""))


def main() -> None:
    comments = parse_comments()
    response = RESPONSE.read_text(encoding="utf-8")

    r1_sections, r2_sections = split_response_sections(response)

    rows = []
    for c in comments:
        pool = r1_sections if c["reviewer"] == 1 else r2_sections
        body = pool[c["n"] - 1] if c["n"] - 1 < len(pool) else ""
        low = body.lower()
        promises = sorted({p.strip() for p in PROMISE_MARKERS if p in low})
        rows.append({
            "id": c["id"],
            "addressed": bool(body.strip()),
            "response_chars": len(body.strip()),
            "promise_phrases": promises,
            "is_promise": bool(promises),
            "comment_head": c["text"][:100].replace("\n", " "),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "response_coverage_audit.csv", index=False)

    print("=" * 100)
    print("Response-letter coverage vs merged reviewer comments")
    print("=" * 100)
    print(f"comments parsed: {len(comments)}  (R1: {sum(c['reviewer']==1 for c in comments)}, "
          f"R2: {sum(c['reviewer']==2 for c in comments)})")
    print(f"response sections with a body: {int(df.addressed.sum())}/{len(df)}")
    print()
    print(f"{'id':<6} {'chars':>6} {'promise?':<9} comment")
    print("-" * 100)
    for r in df.itertuples(index=False):
        flag = "YES" if r.is_promise else ""
        print(f"{r.id:<6} {r.response_chars:>6} {flag:<9} {r.comment_head}")
    print()
    still = df[df.is_promise]
    print(f"sections still containing forward-looking promise wording: {len(still)}")
    for r in still.itertuples(index=False):
        print(f"  {r.id}: {', '.join(r.promise_phrases)}")
    print()
    print("unaddressed comments:", list(df[~df.addressed].id) or "none")

    (OUT / "response_coverage_audit.json").write_text(
        json.dumps({"n_comments": len(comments), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
