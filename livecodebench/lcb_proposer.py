"""The PROPOSER for the LiveCodeBench (code) domain: agentic Claude-Code-on-flash generators + a judge,
each an isolated hard-killed subprocess. Mechanics mirror text_to_sql/proposer.py; prompts are rewritten
for competitive-programming code. The label-free signal the proposer reads = PUBLIC sample-test results
(hidden tests are never shown). Run as a worker:  python -m ...lcb_proposer --worker|--picker <task.json>
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import lcb_bridge as bridge
from text_to_sql import claude_wrapper
from .lcb_common import PKG, PKG_DIR, AGENTS_DIR, MH_ROOT

PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]


def proposal_card_path(run_dir, candidate):
    """Location of the immutable, public-only lineage card for a candidate."""
    return Path(run_dir) / "proposal_cards" / f"{candidate}.json"


def valid_mmh_proposal_card(path, candidate, base_candidate, branch_id, generation_round,
                            peer_candidates):
    """Validate the MMH card without importing the memory package or LCB bridge."""
    try:
        from .mmh_adapter import valid_mmh_proposal_card as validate
        card = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ImportError):
        return False
    return validate(card, candidate=candidate, parent=base_candidate, branch_id=branch_id,
                    generation_round=generation_round, peer_candidates=peer_candidates)


def desc(res):
    if not res.get("ok"):
        return "ERROR: " + str(res.get("error", ""))[:120]
    if not res.get("rows"):
        return "(no output)"
    return ("; ".join(str(r) for r in res["rows"][:3]))[:160]


def _check_script(new_name, batch_json):
    """A standalone script the generator runs to VERIFY its harness on the batch (public tests only)."""
    return (
        "import json\n"
        f"from {PKG} import lcb_bridge as bridge\n"
        f"from {PKG}.lcb_common import load_harness\n"
        f"batch = json.load(open({json.dumps(batch_json)}, encoding='utf-8'))\n"
        "probs = {p.qid: p for p in bridge.load_problems('test6', stdin_only=False)}\n"
        "ok = 0\n"
        "for i, it in enumerate(batch):\n"
        "    p = probs[it['qid']]\n"
        "    try:\n"
        f"        code = load_harness({json.dumps(new_name)}, p).solve()\n"
        "        res = bridge.run_code(code, p.public_tests, starter_code=p.starter_code)\n"
        "        allp = res['n_total'] > 0 and res['n_pass'] == res['n_total']\n"
        "        ok += allp\n"
        "        print('Q%d [%s/%s] public=%d/%d%s code_len=%d' % (i, it['qid'], it.get('difficulty','?'),\n"
        "              res['n_pass'], res['n_total'], ' ALL' if allp else '', len(code)))\n"
        "    except Exception:\n"
        "        import traceback; print('Q%d HARNESS-CRASH' % i); traceback.print_exc()\n"
        "print('SUMMARY: %d/%d pass ALL public tests' % (ok, len(batch)))\n")


def propose_batch(tk):
    """ONE GENERATOR session (branch gi): deep-read ALL peers' traces for context, but IMPROVE only its
    assigned base harness -> write agents/<new_name>.py. Worker subprocess."""
    candidates, trace_dir = tk["candidates"], tk["trace_dir"]
    new_name, gi, tag, run_dir = tk["new_name"], tk["gi"], tk["tag"], tk["run_dir"]
    base_candidate = tk.get("base_candidate", candidates[0])
    if base_candidate not in candidates:
        raise ValueError(f"branch base {base_candidate!r} is not in active candidates")
    cand_path = AGENTS_DIR / f"{new_name}.py"
    base_path = AGENTS_DIR / f"{base_candidate}.py"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    check_path = Path(run_dir) / f"check_{tag}_g{gi}.py"
    mmh_mode = bool(tk.get("memory_mode"))
    card_path = proposal_card_path(run_dir, new_name)
    expected_peer_candidates = [candidate for candidate in candidates if candidate != base_candidate]
    cand_path.unlink(missing_ok=True)
    if mmh_mode:
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.unlink(missing_ok=True)
    shutil.copyfile(base_path, cand_path)
    check_path.write_text(_check_script(new_name, tk["batch_json"]))
    cand_rows = []
    for c in candidates:
        relation = "ASSIGNED BASE" if c == base_candidate else "PEER EVIDENCE"
        cand_rows.append(f"  - `{c}` [{relation}]  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)")
    cand_list = "\n".join(cand_rows)
    memory_section = ""
    if mmh_mode:
        memory_section = (
            "\n\nMETA-MEMORY (PUBLIC-ONLY, ADVISORY): these retrieved rules were selected from earlier "
            "public-test execution. They guide your ONE intervention but do not prove correctness. Do not put "
            "them into the frozen coder prompt automatically; use them only to decide the harness edit.\n"
            f"{json.dumps(tk.get('memory_context', []), indent=2)}\n\n"
            "MMH ARTIFACT (mandatory): write a JSON proposal card to "
            f"`{card_path}` with EXACTLY these fields: `candidate` (exactly `{new_name}`), `branch_id` "
            f"(integer `{gi}`), `generation_round` (exactly `{tag}`), `base_candidate` (exactly "
            f"`{base_candidate}`), `peer_candidates` (exactly {json.dumps(expected_peer_candidates)}), `role` "
            "(string), `behavior_changes` (a list containing EXACTLY ONE object describing the single "
            "intervention), `applied_rule` (object with `rule_id`, `instruction`, and optional `scope`), and "
            "`memory_patch` (object with `operation` ADD|DELETE|REFINE|SPLIT|MERGE, `target_rule_ids` list, "
            "`result_rules` list, `judge_confidence`, `rationale`, and optional general `context`). ADD has an empty target list; all other "
            "operations name existing target rule IDs. DELETE has no result rules; ADD/REFINE/MERGE have one; "
            "SPLIT has at least two. Every result rule has `phi`, `psi`, `omega`, and numeric `confidence`). The card "
            "must describe only public traces and self-generated checks. Never include hidden/private/gold data "
            "or scores. Do not make compound changes: they cannot be independently validated.\n"
        )
    prompt = (
        f"You are evolving a GENERAL competitive-programming HARNESS: arbitrary Python wrapping a FROZEN weak "
        f"coder (deepseek-flash). There are {len(candidates)} candidate harnesses; each was run on a BATCH of "
        f"problems and its FULL trace saved. You CANNOT see hidden tests — only the PUBLIC sample tests "
        f"(label-free).\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"BRANCH LINEAGE — this is mandatory:\n"
        f"- You are branch G{gi}. Your assigned base is `{base_candidate}`.\n"
        f"- `{cand_path}` has already been copied byte-for-byte from that base. Edit this copy in place.\n"
        f"- Other candidates and their traces are PEER EVIDENCE: learn from and test their mechanisms, but do "
        f"NOT switch your parent or replace your harness wholesale with a peer harness. The output must remain a "
        f"descendant of `{base_candidate}`; if the evidence is insufficient, make a minimal or no-op change "
        f"rather than adopting another branch.\n\n"
        f"DEEP-READ THE FULL TRACES (most important step). Each `{trace_dir}/<harness>__q<j>.md` has, for ONE "
        f"harness on ONE problem: the PROBLEM statement, every coder call (its prompt, response, and whether "
        f"THINKING was on/off), the FINAL code, and the PUBLIC-TEST RESULTS (per test: input, expected, got, "
        f"pass/fail). A harness that passes MORE public tests is better. You may READ THE SOURCE of any peer at "
        f"`{AGENTS_DIR}/<name>.py` to understand a mechanism, but your implementation base remains "
        f"`{base_candidate}` — port only a small general mechanism when its trace evidence supports it. IMPROVE "
        f"your assigned base so it passes MORE public tests across the batch.\n\n"
        f"There is NO prescribed recipe and no menu of techniques — the design is ENTIRELY yours. The traces are "
        f"your only evidence of how each harness actually behaves: where the coder is called (and with what "
        f"thinking setting), what it returns, whether the final code is empty or complete, and which public "
        f"tests pass or fail and why. Read them, diagnose the failures YOURSELF, and decide what (if anything) to "
        f"change. Infer what helps from the trace evidence, not from assumptions.\n\n"
        f"Harness API — these primitives EXIST; WHETHER, WHEN and HOW to use them is entirely up to you to "
        f"decide from the traces: self.content (problem text); self.public_tests (list of {{input,output}}); "
        f"self.starter_code; self.llm(prompt, system='', thinking=False|'low'|'medium'|'high', n=1) -> the "
        f"frozen coder; self.run_public(code) -> {{n_pass,n_total,results}}; self.stress(code) -> runs code on "
        f"self-generated max-constraint inputs; bridge.extract_code(text); bridge.back_translate(code). "
        f"GENERAL — never hardcode a problem's answer; NEVER touch the hidden tests (inputs or outputs) or "
        f"is_correct; no per-problem special-casing. Keep `from ..harness_base import CodeHarness` and "
        f"`from .. import lcb_bridge as bridge`. Do NOT write infinite loops / catastrophic regex (they HANG).\n\n"
        f"VERIFY: edit `{cand_path}`, then run  PYTHONPATH={MH_ROOT} python {check_path}  (prints per-problem "
        f"public pass-counts + a SUMMARY); iterate to RAISE the count without breaking others. Leave the final "
        f"harness at EXACTLY `{cand_path}`."
        + memory_section
        + (f"\n\nDIVERSITY (one of several parallel attempts): {tk.get('diversity','')}" if tk.get("diversity") else ""))
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(run_dir) / "claude_sessions"), name=f"{tag}g{gi}",
                       timeout_seconds=tk["timeout"], progress=False)
    return cand_path.exists() and (
        not mmh_mode or valid_mmh_proposal_card(card_path, new_name, base_candidate, gi, tag,
                                                 expected_peer_candidates)
    )


# Three SEARCH STANCES (not technique recipes) — each branch explores the harness space from a different
# angle, so the batch covers more ground. They say HOW to search, never WHICH mechanism to write.
DIVERSITY = [
    (
        "CONSERVATIVE ROLE: preserve every parent behavior that has sound trace evidence. Make the smallest "
        "general change that fixes a concrete failure you can actually see in the traces. Do not over-generalize "
        "one problem's fix into a universal rule."
    ),
    (
        "EXPLORATION ROLE: test a genuinely different hypothesis about WHY the harness fails than the "
        "conservative branch would. Implement the simplest general mechanism your hypothesis implies; do not "
        "copy another branch's mechanism merely because it is the current default."
    ),
    (
        "ADVERSARIAL ROLE: start from the disagreements among the existing candidates and the problems where "
        "they fail. Scrutinize what each harness ACTUALLY does in the traces before trusting it; never assume a "
        "harness is correct just because it passes some public tests. Keep only what the evidence supports."
    ),
]


def _terminate_process_tree(process):
    """Hard-stop an agent worker and descendants on both supported platforms."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        # ``start_new_session`` does not give Windows a POSIX process group;
        # taskkill /T is the equivalent tree termination required for a hung
        # Claude subprocess and any children it launched.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _spawn_and_wait(procs, timeout):
    hard = time.time() + timeout + 120
    for p in procs:
        try:
            p.wait(timeout=max(1, hard - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            _terminate_process_tree(p)


def sample_branches(branches, trace_dir, run_dir, tag, run_name, batch, model, timeout,
                    memory_mode=False, memory_context=None):
    """Advance fixed proposer branches once.

    Every branch sees the same active candidate--trace pairs, while branch ``gi`` receives ``branches[gi]``
    as its mechanically pre-copied edit base. The returned list preserves branch order and uses ``None`` for
    a failed child."""
    active = list(dict.fromkeys(branches))
    batch_json = Path(run_dir) / f"batch_{tag}.json"
    batch_json.write_text(json.dumps([{"qid": p.qid, "difficulty": getattr(p, "difficulty", "?")} for p in batch]))
    names = [f"cand_{run_name}_{tag}_g{gi}" for gi in range(len(branches))]
    procs = []
    for gi, (base_candidate, name) in enumerate(zip(branches, names)):
        (AGENTS_DIR / f"{name}.py").unlink(missing_ok=True)
        task = {"kind": "generate_batch", "candidates": active, "base_candidate": base_candidate,
                "trace_dir": str(trace_dir), "batch_json": str(batch_json), "new_name": name,
                "run_dir": str(run_dir), "tag": tag, "gi": gi, "model": model, "timeout": timeout,
                "diversity": DIVERSITY[gi % len(DIVERSITY)], "memory_mode": bool(memory_mode),
                "memory_context": memory_context or []}
        tpath = Path(run_dir) / f"task_{tag}_g{gi}.json"
        tpath.write_text(json.dumps(task))
        procs.append(subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.lcb_proposer", "--worker", str(tpath)],
                                      cwd=str(MH_ROOT), start_new_session=True,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    _spawn_and_wait(procs, timeout)
    return [
        name if (
            (AGENTS_DIR / f"{name}.py").exists()
            and (not memory_mode or valid_mmh_proposal_card(
                proposal_card_path(run_dir, name), name, branches[gi], gi, tag,
                [candidate for candidate in active if candidate != branches[gi]],
            ))
        ) else None
        for gi, name in enumerate(names)
    ]


def run_picker(tk):
    """ONE JUDGE session: deep-read all candidates' traces and INVESTIGATE which harness is genuinely correct
    on the most problems. It is given tools, not a fixed metric — mirrors the SQL judge's 'probe, don't
    eyeball' stance (LCB has no gold Hint, so investigation is the only anchor)."""
    cands, trace_dir, choice_path = tk["candidates"], tk["trace_dir"], tk["choice_path"]
    incumbent = tk.get("incumbent") or cands[0]
    cand_list = "\n".join(f"  - `{c}`  (traces: {trace_dir}/{c}__q*.md ; code: {AGENTS_DIR}/{c}.py)" for c in cands)
    prompt = (
        f"You are the JUDGE. There are {len(cands)} candidate competitive-programming harnesses; each ran on a "
        f"BATCH of problems with its FULL trace saved. You do NOT write/change harnesses — pick the SINGLE "
        f"harness that is GENUINELY CORRECT on the MOST problems in the batch. You CANNOT see the hidden tests.\n\n"
        f"CANDIDATES:\n{cand_list}\n\n"
        f"Each `{trace_dir}/<harness>__q<j>.md` has the PROBLEM, every coder call (+ thinking on/off), the final "
        f"code, and the PUBLIC-TEST RESULTS.\n\n"
        f"THE TRAP YOU MUST AVOID: passing the PUBLIC samples does NOT mean correct. The public samples are tiny "
        f"(often 1-3 cases) and non-adversarial; a plausible-but-wrong program passes them and still fails the "
        f"hidden suite. Measured on this exact task, ~29% of public-passing solutions are hidden-WRONG. So DO NOT "
        f"rank by public-pass count — that metric systematically rewards these false positives. Public-pass is a "
        f"NECESSARY filter (a solution failing public is out), not proof of correctness.\n\n"
        f"YOUR JOB IS TO INVESTIGATE, not to tally. You have a shell (Bash) and the full bridge. On the problems "
        f"where the candidates DIFFER, dig in and find which one is actually right, using any label-free evidence "
        f"you can generate yourself — you are ENCOURAGED to build your own checks:\n"
        f"  - RE-RUN each candidate's code on the public samples first (filter). PYTHONPATH={MH_ROOT} python -c "
        f"\"from {PKG} import lcb_bridge as b; from {PKG}.lcb_common import load_harness; "
        f"probs={{p.qid:p for p in b.load_problems('test6',stdin_only=False)}}; p=probs['<qid>']; "
        f"c=load_harness('<name>',p).solve(); print(b.run_code(c,p.public_tests,starter_code=p.starter_code))\".\n"
        f"  - DIFFERENTIAL TEST: for a problem, run the public-passing candidates' programs on the SAME self-made "
        f"inputs (b.gen_stress_inputs(p) gives legal max-constraint inputs; or write your own valid inputs) and "
        f"compare their outputs. If two programs disagree on a legal input, at least one is WRONG — the odd one "
        f"out is the prime suspect. For a unique-answer problem, the truly-correct programs must all agree.\n"
        f"  - ROBUSTNESS: b.run_stress(code, b.gen_stress_inputs(p)) flags crash/TLE/empty on large inputs — a "
        f"public-passing solution that crashes there is almost certainly hidden-wrong.\n"
        f"  - INTENT: b.back_translate(code) says in English what the code actually computes; compare to the "
        f"problem statement — a mismatch is a red flag.\n\n"
        f"CAVEATS (do not be naive): the coder's errors are often CORRELATED — several independent programs can be "
        f"identically WRONG, so agreement is a weak prior, NOT proof; never decide by majority vote alone. And a "
        f"self-generated input can be malformed — if a program you believe is correct crashes on it, suspect the "
        f"INPUT, not the program. Base each verdict on a check you actually ran, and prefer the harness with the "
        f"most problems that survive your scrutiny. Never touch the hidden tests or any answer key. True ties -> "
        f"the simpler / more general harness.\n\n"
        f"THE INCUMBENT. `{incumbent}` is the harness currently in force — every other candidate is a "
        f"CHALLENGER descended from it. The burden of proof is on the challenger: keep `{incumbent}` unless "
        f"you can point to specific problems where a challenger is verifiably right and it is wrong. A tie "
        f"goes to the incumbent. Evolution that cannot demonstrate an improvement should not be adopted.\n"
        f"NEVER break a tie on ARCHITECTURE — 'more robust regex', 'better system prompt', 'looks more "
        f"general' describe SOURCE CODE, not OUTPUTS, and are how a judge talks itself into the wrong "
        f"answer. If candidates are tied on measured results, find another problem to discriminate on.\n"

        f"Write ONLY the chosen harness's exact NAME to `{choice_path}` (run:  echo <name> > {choice_path} ). Then STOP.")
    claude_wrapper.run(prompt=prompt, model=tk["model"], allowed_tools=PROPOSER_TOOLS, cwd=str(MH_ROOT),
                       log_dir=str(Path(tk["run_dir"]) / "claude_sessions"), name=f"judge_{tk['tag']}",
                       timeout_seconds=tk["timeout"], progress=False)


def pick_batch(candidates, trace_dir, run_dir, tag, model, timeout, incumbent=None):
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    choice_path = Path(run_dir) / f"judge_{tag}.txt"
    choice_path.unlink(missing_ok=True)
    tpath = Path(run_dir) / f"judgetask_{tag}.json"
    tpath.write_text(json.dumps({"candidates": candidates, "trace_dir": str(trace_dir),
                                 "choice_path": str(choice_path), "run_dir": str(run_dir), "tag": tag,
                                 "model": model, "timeout": timeout,
                                 "incumbent": incumbent or candidates[0]}))
    p = subprocess.Popen([sys.executable, "-u", "-m", f"{PKG}.lcb_proposer", "--picker", str(tpath)],
                         cwd=str(MH_ROOT), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _spawn_and_wait([p], timeout)
    if choice_path.exists():
        nm = choice_path.read_text().strip().split()
        if nm and nm[0] in candidates:
            return nm[0]
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--picker")
    a = ap.parse_args()
    if a.picker:
        run_picker(json.load(open(a.picker)))
    else:
        propose_batch(json.load(open(a.worker)))
