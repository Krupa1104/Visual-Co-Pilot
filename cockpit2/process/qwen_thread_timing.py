r"""
qwen_thread_timing.py -- measure Qwen routing time on this PC, and show Qwen's raw output.

Run 1 (4 Nov) ruled out the "Gradio worker thread" idea: same torch thread count (14) in both
threads, and the same question took 5-14 s. This version checks the next candidates:
  * torch CPU thread count (on CPUs with performance + efficiency cores, using fewer threads can be
    faster), tried at several values, 2 repeats per question;
  * the raw text Qwen generates (to see why "Are we airborne?" gave PARSE_FAILED).
It changes nothing in the app; it only measures.

Run from the repo root (venv_final), with the app CLOSED and the laptop on charger:
  python cockpit2\process\qwen_thread_timing.py
"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COCKPIT2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(COCKPIT2, "gui"))

import torch  # noqa: E402
from copilot_core_v2 import CopilotSystemV2  # noqa: E402

QUESTIONS = ["Is the plane going up?", "Which way is the aircraft facing?", "Are we airborne?"]
THREADS = [4, 6, 8, torch.get_num_threads()]

c = CopilotSystemV2(model_dir=os.path.join(COCKPIT2, "dataset_v2", "model_v2"),
                    use_llm=True, rules_first=False)
if not c.have_llm_gate:
    sys.exit("Qwen could not be loaded.")


def raw_qwen(question):
    """Same steps as CopilotSystemV2._llm_classify, but returns Qwen's raw text."""
    msgs = [{"role": "system", "content": c._sys_prompt}, {"role": "user", "content": question}]
    prompt = c._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = c._tok(prompt, return_tensors="pt").to(c.llm_device)
    with torch.no_grad():
        out = c._llm.generate(**inputs, max_new_tokens=8, do_sample=False,
                              pad_token_id=c._tok.eos_token_id)
    text = c._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, inputs["input_ids"].shape[1]


print(f"os.cpu_count() = {os.cpu_count()}, default torch threads = {torch.get_num_threads()}")
text, n_tok = raw_qwen("warm-up")
print(f"prompt length = {n_tok} tokens")

for n in THREADS:
    torch.set_num_threads(n)
    times = []
    print(f"\n--- torch threads = {n} ---")
    for q in QUESTIONS:
        for rep in range(2):
            t0 = time.perf_counter()
            text, _ = raw_qwen(q)
            dt = time.perf_counter() - t0
            times.append(dt)
            if rep == 0:
                print(f"  {q!r}: raw Qwen output = {text!r}")
    print(f"  time per question: median {statistics.median(times):.2f} s, "
          f"min {min(times):.2f} s, max {max(times):.2f} s")