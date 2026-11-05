r"""
eval_router.py -- compare the cockpit question routers on a fixed question set.

Routers compared (all on the same questions):
  keyword          CopilotSystemV2.keyword_fallback (substring matching, the old last-resort tier)
  rules            rule_route() alone (None = "not answered by rules")
  qwen             Qwen2.5-1.5B gate alone (the current app behaviour)
  rules_then_qwen  rule_route(), and Qwen only for questions the rules leave (adopted 4 Nov)
  qwen_choice      Qwen picks one of 9 numbered options from next-token scores (llm_mode="choice");
                   an option with probability < 0.35 counts as "not in scope", as in the app
  rules_then_qwen_choice   rule_route(), then qwen_choice for the rest

Qwen is run once per question and its label/time reused for "rules_then_qwen" (generation is greedy,
do_sample=False, so a second run would give the same label).

Decision rules, fixed BEFORE running (4 Nov 2026):
  1. rules_then_qwen replaces qwen if its overall accuracy is not lower than qwen's AND its median
     routing time is lower. Otherwise the app keeps qwen-only routing.
  2. (added 4 Nov, before the first choice-mode run) llm_mode="choice" replaces "generate" if
     rules_then_qwen_choice accuracy is not lower than rules_then_qwen AND qwen_choice's median time
     is not higher than qwen's. Decide on a question set NOT used to design the change.

Run from the repo root (needs the cockpit model, because CopilotSystemV2 loads it):
  python cockpit2\process\eval_router.py
Output: cockpit2\results\router_eval\per_question.csv and summary.json
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COCKPIT2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(COCKPIT2, "gui"))

import torch  # noqa: E402  (before anything that could import paddle, see integrated_final.py)
from copilot_core_v2 import OUT_OF_SCOPE, CopilotSystemV2, rule_route  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=os.path.join(HERE, "router_questions.csv"))
ap.add_argument("--model_dir", default=os.path.join(COCKPIT2, "dataset_v2", "model_v2"))
ap.add_argument("--out_dir", default=None,
                help="default: cockpit2/results/router_eval/<questions file name>/")
ap.add_argument("--llm_on_cpu", action="store_true")
ap.add_argument("--torch_threads", type=int, default=None, help="torch CPU threads (default: torch's)")
ap.add_argument("--choice_threshold", type=float, default=0.35, help="app's routing threshold")
args = ap.parse_args()
if args.out_dir is None:
    args.out_dir = os.path.join(COCKPIT2, "results", "router_eval",
                                os.path.splitext(os.path.basename(args.questions))[0])

if args.torch_threads:
    torch.set_num_threads(args.torch_threads)
with open(args.questions, newline="", encoding="utf-8") as f:
    rows = [{"question": r["question"], "expected": None if r["expected"] == "none" else r["expected"]}
            for r in csv.DictReader(f)]

sysm = CopilotSystemV2(model_dir=args.model_dir, use_llm=True, rules_first=False, llm_on_cpu=args.llm_on_cpu)
if not sysm.have_llm_gate:
    sys.exit("Qwen could not be loaded -- see the warning above. Nothing evaluated.")
sysm._llm_classify("warm-up question")   # first generate() call is slower; keep it out of the timings
have_choice = getattr(sysm, "_choice_token_ids", None) is not None
if have_choice:
    sysm._llm_choose("warm-up question")

out = []
for i, r in enumerate(rows, 1):
    q = r["question"]
    t0 = time.perf_counter(); kw = sysm.keyword_fallback(q); kw_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); ru = rule_route(q); ru_ms = (time.perf_counter() - t0) * 1000
    ru_label = None if ru == OUT_OF_SCOPE else ru     # OUT_OF_SCOPE = rules answered "not in scope"
    t0 = time.perf_counter()
    qw = sysm._llm_classify(q)
    if qw == "PARSE_FAILED":           # same handling as route_question()
        qw = sysm.keyword_fallback(q)
    qw_ms = (time.perf_counter() - t0) * 1000
    rtq, rtq_ms = (ru_label, ru_ms) if ru is not None else (qw, ru_ms + qw_ms)
    if have_choice:
        t0 = time.perf_counter()
        qc, qc_p = sysm._llm_choose(q)
        qc_ms = (time.perf_counter() - t0) * 1000
        qc_app = qc if (qc is not None and qc_p >= args.choice_threshold) else None   # app behaviour
        rtc, rtc_ms = (ru_label, ru_ms) if ru is not None else (qc_app, ru_ms + qc_ms)
    else:
        qc = qc_app = rtc = None; qc_p = qc_ms = rtc_ms = float("nan")
    out.append({"question": q, "expected": r["expected"] or "none",
                "keyword": kw or "none", "keyword_ms": kw_ms,
                "rules": "(deferred)" if ru is None else (ru_label or "none"), "rules_ms": ru_ms,
                "qwen": qw or "none", "qwen_ms": qw_ms,
                "rules_then_qwen": rtq or "none", "rules_then_qwen_ms": rtq_ms,
                "rules_then_qwen_used": "rules" if ru is not None else "qwen",
                "qwen_choice": qc_app or "none", "qwen_choice_raw": qc or "none", "qwen_choice_p": qc_p,
                "qwen_choice_ms": qc_ms, "rules_then_qwen_choice": rtc or "none",
                "rules_then_qwen_choice_ms": rtc_ms})
    print(f"[{i}/{len(rows)}] {q!r}: qwen={qw} ({qw_ms / 1000:.1f} s) rules={ru} -> {rtq}", flush=True)


def summarise(name):
    ms = [o[f"{name}_ms"] for o in out]
    if name == "rules":
        answered = [o for o in out if o["rules"] != "(deferred)"]
        return {"answered": len(answered), "of": len(out),
                "correct_when_answered": sum(o["rules"] == o["expected"] for o in answered)}
    correct = sum(o[name] == o["expected"] for o in out)
    ins = [o for o in out if o["expected"] != "none"]
    oos = [o for o in out if o["expected"] == "none"]
    return {"accuracy": correct / len(out), "correct": correct, "of": len(out),
            "in_scope_correct": sum(o[name] == o["expected"] for o in ins), "in_scope": len(ins),
            "off_topic_rejected": sum(o[name] == "none" for o in oos), "off_topic": len(oos),
            "median_ms": statistics.median(ms), "mean_ms": statistics.mean(ms), "max_ms": max(ms)}


names = ["keyword", "rules", "qwen", "rules_then_qwen"] + (["qwen_choice", "rules_then_qwen_choice"]
                                                           if have_choice else [])
summary = {n: summarise(n) for n in names}
summary["rules_then_qwen"]["answered_by_rules"] = sum(o["rules_then_qwen_used"] == "rules" for o in out)
q, r = summary["qwen"], summary["rules_then_qwen"]
summary["decision"] = ("adopt rules_then_qwen" if r["accuracy"] >= q["accuracy"] and r["median_ms"] < q["median_ms"]
                       else "keep qwen-only")
if have_choice:
    c, g = summary["rules_then_qwen_choice"], summary["rules_then_qwen"]
    summary["decision_llm_mode"] = ("adopt choice" if c["accuracy"] >= g["accuracy"]
                                    and summary["qwen_choice"]["median_ms"] <= q["median_ms"]
                                    else "keep generate")
summary["setup"] = {"torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                    "device": str(sysm.llm_device), "questions": args.questions}

os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
