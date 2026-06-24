"""
eval.py

Evaluate retrieval and generation quality against eval_set.json.

    python eval.py           # retrieval eval only
    python eval.py --gen     # retrieval + LLM-as-judge generation eval

Retrieval:
  True positives: expected_section must appear in the retrieved chunk metadata.
  True negatives: no chunks should survive the similarity threshold.

Generation (--gen):
  Calls answer_conceptual_question() for each true positive, then uses a
  Claude judge to score the answer 1-5 against the reference answer.
"""

import argparse
import json
from pathlib import Path
from rag import retrieve, answer_conceptual_question, get_client, CLAUDE_MODEL, _SIMILARITY_THRESHOLD

_JUDGE_SYSTEM = (
    "You are evaluating a discrete mathematics tutoring system. "
    "Score the student-facing answer on a 1–5 scale:\n"
    "  5 — fully correct and complete\n"
    "  4 — correct with minor omissions\n"
    "  3 — partially correct, key idea present but incomplete or imprecise\n"
    "  2 — mostly wrong but contains a relevant element\n"
    "  1 — wrong or misleading\n\n"
    "Reply with exactly two lines:\n"
    "SCORE: <number>\n"
    "REASON: <one sentence>"
)


def _judge(question: str, reference: str, answer: str) -> tuple[int, str]:
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Answer to evaluate: {answer}"
    )
    client = get_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=128,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text.strip()
    score, reason = 0, ""
    for line in text.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


def run_retrieval_eval(items: list[dict], k: int) -> list[dict]:
    tp_results = []
    tn_results = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        print(f"\r  [{idx}/{total}] retrieving...", end="", flush=True)
        chunks = retrieve(item["question"], k=k, threshold=0.0)
        section_scores = {c["metadata"].get("section"): c["score"] for c in chunks}
        above = {s: sc for s, sc in section_scores.items() if sc >= _SIMILARITY_THRESHOLD}

        if item["negative"]:
            tn_results.append({
                "question": item["question"],
                "passed": len(above) == 0,
                "scores": section_scores,
            })
        else:
            expected = item["expected_section"]
            tp_results.append({
                "question": item["question"],
                "expected": expected,
                "passed": expected in above,
                "expected_score": section_scores.get(expected),
                "above": above,
                "reference_answer": item.get("reference_answer"),
            })

    print()

    tp_passed = sum(r["passed"] for r in tp_results)
    print(f"\n{'='*60}")
    print(f"TRUE POSITIVES  {tp_passed}/{len(tp_results)} ({100 * tp_passed // len(tp_results)}%)")
    print(f"{'='*60}")
    for r in tp_results:
        mark = "PASS" if r["passed"] else "FAIL"
        score_str = f"  (score: {r['expected_score']:.3f})" if r["expected_score"] is not None else "  (not retrieved)"
        print(f"  [{mark}] §{r['expected']:5s}{score_str}  {r['question']}")
        if not r["passed"]:
            retrieved_str = ", ".join(f"§{s}={sc:.3f}" for s, sc in sorted(r["above"].items())) or "(none above threshold)"
            print(f"           retrieved: {retrieved_str}")

    tn_passed = sum(r["passed"] for r in tn_results)
    print(f"\n{'='*60}")
    print(f"TRUE NEGATIVES  {tn_passed}/{len(tn_results)} ({100 * tn_passed // len(tn_results)}%)")
    print(f"{'='*60}")
    for r in tn_results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['question']}")
        if not r["passed"]:
            above = {s: sc for s, sc in r["scores"].items() if sc >= _SIMILARITY_THRESHOLD}
            retrieved_str = ", ".join(f"§{s}={sc:.3f}" for s, sc in sorted(above.items()))
            print(f"           retrieved: {retrieved_str}")

    total_passed = tp_passed + tn_passed
    print(f"\n{'='*60}")
    print(f"OVERALL  {total_passed}/{len(items)} ({100 * total_passed // len(items)}%)")
    print(f"  True positives : {tp_passed}/{len(tp_results)}")
    print(f"  True negatives : {tn_passed}/{len(tn_results)}")
    print(f"{'='*60}")

    tp_scores = [r["expected_score"] for r in tp_results if r["expected_score"] is not None]
    tn_max_scores = [max(r["scores"].values()) for r in tn_results if r["scores"]]
    if tp_scores and tn_max_scores:
        min_tp, max_tn = min(tp_scores), max(tn_max_scores)
        print(f"\nCALIBRATION (current threshold: {_SIMILARITY_THRESHOLD:.2f})")
        print(f"  Lowest TP score  : {min_tp:.3f}")
        print(f"  Highest TN score : {max_tn:.3f}")
        if max_tn < min_tp:
            print(f"  Suggested threshold: {round((min_tp + max_tn) / 2, 2):.2f}")
        else:
            print(f"  WARNING: scores overlap — no clean threshold exists.")
    print()

    return tp_results


def run_generation_eval(tp_results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"GENERATION EVAL  (LLM-as-judge, 1–5)")
    print(f"{'='*60}")

    scores = []
    for idx, r in enumerate(tp_results, 1):
        print(f"\r  [{idx}/{len(tp_results)}] judging...", end="", flush=True)
        answer = answer_conceptual_question(r["question"])
        score, reason = _judge(r["question"], r["reference_answer"], answer)
        scores.append(score)
        mark = "✓" if score >= 4 else ("~" if score == 3 else "✗")
        print(f"\r  [{mark}] {score}/5  §{r['expected']:5s}  {r['question']}")
        print(f"         {reason}")

    print(f"\n{'='*60}")
    avg = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= 4)
    print(f"  Average score : {avg:.2f}/5")
    print(f"  Score ≥ 4     : {passed}/{len(scores)} ({100 * passed // len(scores)}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Run generation eval after retrieval eval")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--eval-path", default="eval_set.json")
    args = parser.parse_args()

    items = json.loads(Path(args.eval_path).read_text())
    tp_results = run_retrieval_eval(items, k=args.k)

    if args.gen:
        tp_with_ref = [r for r in tp_results if r.get("reference_answer")]
        run_generation_eval(tp_with_ref)
