"""Per-step loop and repeat taxonomy labeling for normalized agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

from agent_loops.canonicalize import NormalizedStep

PATHOLOGICAL_LOOP_LABELS: frozenset[str] = frozenset(
    {
        "exact_loop",
        "near_duplicate",
        "aba_thrash",
        "period_k_rotation",
        "error_streak",
        "redundant_reread",
    }
)

ALL_STEP_LABELS: tuple[str, ...] = (
    "aba_thrash",
    "period_k_rotation",
    "exact_loop",
    "error_streak",
    "near_duplicate",
    "redundant_reread",
    "productive_repeat",
    "normal",
)

# Primary label priority (first match wins for mutually-exclusive step shares)
_PRIMARY_PRIORITY: tuple[str, ...] = (
    "aba_thrash",
    "period_k_rotation",
    "exact_loop",
    "error_streak",
    "near_duplicate",
    "redundant_reread",
    "productive_repeat",
)


@dataclass(slots=True)
class StepLabel:
    """Labels assigned to one trajectory step."""

    step_idx: int
    labels: set[str] = field(default_factory=set)
    primary_label: str = "normal"
    is_loop: bool = False
    rotation_k: int | None = None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _obs_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[A-Za-z0-9_./:-]+", text[:1500].lower()))


def label_trajectory(steps: list[NormalizedStep]) -> list[StepLabel]:
    """Label every step in a trajectory with loop and repeat categories.

    Labels:
      - exact_loop: same canonical call & normalized obs as an earlier step
      - near_duplicate: highly similar call (different canonical hash) within recent window
        and stalled/identical obs or identical error
      - aba_thrash: A-B-A pattern with unchanged A observation or A-B-A-B alternation
      - period_k_rotation: consecutive period-k cycle (2 <= k <= 5, >= 2 distinct calls)
      - error_streak: same normalized error repeated consecutively
      - redundant_reread: same file (and range when known) read again with no write in between
      - productive_repeat: same canonical call as earlier step, new unseen observation,
        and not classified under any pathological loop label
    """
    n = len(steps)
    results = [StepLabel(step_idx=s.step_idx) for s in steps]
    if n == 0:
        return results

    # Indices of tool-bearing steps so non-tool assistant messages don't break A-B-A / period-k
    tool_positions = [i for i, s in enumerate(steps) if s.has_tool]
    m = len(tool_positions)

    # Track history over tool-bearing steps
    seen_calls: dict[str, list[int]] = {}  # call_hash -> list of tool-pos indices
    seen_pairs: set[tuple[str, str]] = set()  # (call_hash, obs_norm_hash)
    obs_to_tool_pos: dict[str, list[int]] = {}
    err_to_tool_pos: dict[str, list[int]] = {}
    obs_tok_cache: dict[int, set[str]] = {}

    # File read/write state: file_path -> set of read range keys since last write
    unread_or_written_reads: dict[str, set[str]] = {}
    read_obs_since_write: dict[tuple[str, str], str] = {}

    for tp_idx, step_pos in enumerate(tool_positions):
        s = steps[step_pos]
        lbl = results[step_pos]

        # 1. Exact loop & Productive repeat tracking
        if s.args_known and s.obs_known:
            pair = (s.call_hash, s.obs_norm_hash)
            if pair in seen_pairs:
                lbl.labels.add("exact_loop")
            elif s.call_hash in seen_calls:
                # Tentative productive_repeat; cleared later if another loop label fires
                lbl.labels.add("productive_repeat")
            seen_pairs.add(pair)

        # 2. Error streak (consecutive tool-bearing steps with same error_sig)
        if s.is_error and s.error_sig and tp_idx >= 1:
            prev_s = steps[tool_positions[tp_idx - 1]]
            if prev_s.is_error and prev_s.error_sig == s.error_sig:
                lbl.labels.add("error_streak")

        # 3. Redundant re-read (same file read with no write in between)
        if s.file_op == "read" and s.file_path:
            rkey = s.file_range or "full"
            seen_ranges = unread_or_written_reads.setdefault(s.file_path, set())
            if s.read_range_exact:
                if rkey in seen_ranges or ("full" in seen_ranges and rkey != "full"):
                    lbl.labels.add("redundant_reread")
                seen_ranges.add(rkey)
            else:
                # Coarse schema (e.g. TB Read without offset/limit): require matching known obs
                if s.obs_known and (s.file_path, rkey) in read_obs_since_write:
                    if read_obs_since_write[(s.file_path, rkey)] == s.obs_norm_hash:
                        lbl.labels.add("redundant_reread")
                if s.obs_known:
                    read_obs_since_write[(s.file_path, rkey)] = s.obs_norm_hash
        elif s.file_op == "write" and s.file_path:
            unread_or_written_reads.pop(s.file_path, None)
            for k in [k for k in read_obs_since_write if k[0] == s.file_path]:
                del read_obs_since_write[k]
        elif s.file_op == "write_any":
            unread_or_written_reads.clear()
            read_obs_since_write.clear()

        # 4. Near-duplicate check (O(1) via obs/error index + <=3 immediate neighbors)
        if s.args_known and s.obs_known:
            candidates: set[int] = set()
            for cand_tp in obs_to_tool_pos.get(s.obs_norm_hash, ()):
                if 1 <= tp_idx - cand_tp <= 15:
                    candidates.add(cand_tp)
            if s.is_error and s.error_sig:
                for cand_tp in err_to_tool_pos.get(s.error_sig, ()):
                    if 1 <= tp_idx - cand_tp <= 15:
                        candidates.add(cand_tp)
            for back in range(1, min(4, tp_idx + 1)):
                candidates.add(tp_idx - back)

            toks_i = set(s.call_tokens)
            len_i = len(s.canonical_call)
            for cand_tp in sorted(candidates, reverse=True):
                cand_s = steps[tool_positions[cand_tp]]
                if not cand_s.args_known or not cand_s.obs_known:
                    continue
                if cand_s.tool_name != s.tool_name or cand_s.call_hash == s.call_hash:
                    continue
                len_j = len(cand_s.canonical_call)
                if len_i == 0 or len_j == 0:
                    continue
                ratio_len = min(len_i, len_j) / max(len_i, len_j)
                if ratio_len < 0.70:
                    continue
                tok_sim = _jaccard(toks_i, set(cand_s.call_tokens))
                if tok_sim < 0.80:
                    continue
                a_cmp = s.canonical_call if len_i <= 600 else (s.canonical_call[:400] + s.canonical_call[-200:])
                b_cmp = cand_s.canonical_call if len_j <= 600 else (cand_s.canonical_call[:400] + cand_s.canonical_call[-200:])
                char_sim = SequenceMatcher(None, a_cmp, b_cmp).ratio()
                if char_sim < 0.85:
                    continue

                obs_stalled = (
                    s.obs_norm_hash == cand_s.obs_norm_hash
                    or (s.is_error and s.error_sig is not None and s.error_sig == cand_s.error_sig)
                )
                if not obs_stalled and (tp_idx - cand_tp) <= 3:
                    ti = obs_tok_cache.setdefault(step_pos, _obs_tokens(s.obs_text))
                    tj = obs_tok_cache.setdefault(
                        tool_positions[cand_tp], _obs_tokens(cand_s.obs_text)
                    )
                    if ti and tj and _jaccard(ti, tj) >= 0.90:
                        obs_stalled = True

                if obs_stalled:
                    lbl.labels.add("near_duplicate")
                    break

        if s.args_known:
            seen_calls.setdefault(s.call_hash, []).append(tp_idx)
        if s.obs_known:
            obs_to_tool_pos.setdefault(s.obs_norm_hash, []).append(tp_idx)
        if s.is_error and s.error_sig:
            err_to_tool_pos.setdefault(s.error_sig, []).append(tp_idx)

    # 5. A-B-A thrash & Period-k rotation (2 <= k <= 5) over tool-bearing sequence
    for tp_idx in range(m):
        pos_i = tool_positions[tp_idx]
        s_i = steps[pos_i]
        if not s_i.args_known:
            continue

        # A-B-A thrash
        if tp_idx >= 2:
            pos_1 = tool_positions[tp_idx - 1]
            pos_2 = tool_positions[tp_idx - 2]
            s_1, s_2 = steps[pos_1], steps[pos_2]
            if s_1.args_known and s_2.args_known and s_i.call_hash == s_2.call_hash and s_i.call_hash != s_1.call_hash:
                same_obs = s_i.obs_known and s_2.obs_known and s_i.obs_norm_hash == s_2.obs_norm_hash
                is_abab = (
                    tp_idx >= 3
                    and steps[tool_positions[tp_idx - 3]].args_known
                    and s_1.call_hash == steps[tool_positions[tp_idx - 3]].call_hash
                )
                if same_obs or is_abab:
                    results[pos_i].labels.add("aba_thrash")
                    if is_abab:
                        results[pos_1].labels.add("aba_thrash")
                        results[pos_2].labels.add("aba_thrash")

        # Period-k rotation (2 <= k <= 5)
        for k in range(2, 6):
            if tp_idx < 2 * k - 1:
                continue
            block2_pos = tool_positions[tp_idx - k + 1 : tp_idx + 1]
            block1_pos = tool_positions[tp_idx - 2 * k + 1 : tp_idx - k + 1]
            if not all(steps[p].args_known for p in block1_pos + block2_pos):
                continue
            hashes2 = [steps[p].call_hash for p in block2_pos]
            hashes1 = [steps[p].call_hash for p in block1_pos]
            if hashes1 != hashes2 or len(set(hashes2)) < 2:
                continue
            # Ensure primitive period (not a multiple of smaller period, e.g. k=4 matching A,B,A,B)
            is_primitive = True
            for sub_k in range(1, k):
                if k % sub_k == 0 and hashes2 == hashes2[:sub_k] * (k // sub_k):
                    is_primitive = False
                    break
            if not is_primitive:
                continue

            # Guard against productive edit->test->edit->test cycles: require at least one
            # unchanged observation across the cycle, OR no write ops in cycle, OR >= 3 cycles
            any_same_obs = any(
                steps[p2].obs_known
                and steps[p1].obs_known
                and steps[p2].obs_norm_hash == steps[p1].obs_norm_hash
                for p1, p2 in zip(block1_pos, block2_pos)
            )
            has_write = any(
                steps[p].file_op in ("write", "write_any") for p in block1_pos + block2_pos
            )
            has_three_cycles = False
            if tp_idx >= 3 * k - 1:
                block0_pos = tool_positions[tp_idx - 3 * k + 1 : tp_idx - 2 * k + 1]
                if all(steps[p].args_known for p in block0_pos) and [
                    steps[p].call_hash for p in block0_pos
                ] == hashes2:
                    has_three_cycles = True

            if any_same_obs or not has_write or has_three_cycles:
                for p in block2_pos:
                    results[p].labels.add("period_k_rotation")
                    if results[p].rotation_k is None or k < results[p].rotation_k:
                        results[p].rotation_k = k
                break

    # Finalize productive_repeat exclusivity, is_loop, and primary_label
    for lbl in results:
        if lbl.labels & PATHOLOGICAL_LOOP_LABELS:
            lbl.labels.discard("productive_repeat")
            lbl.is_loop = True
        else:
            lbl.is_loop = False
        for cand in _PRIMARY_PRIORITY:
            if cand in lbl.labels:
                lbl.primary_label = cand
                break
        else:
            lbl.primary_label = "normal"

    return results


STRICT_LOOP_LABELS: frozenset[str] = frozenset(
    {
        "exact_loop",
        "aba_thrash",
        "period_k_rotation",
    }
)

_STRICT_PRIORITY: tuple[str, ...] = (
    "aba_thrash",
    "period_k_rotation",
    "exact_loop",
)


def label_trajectory_strict(steps: list[NormalizedStep]) -> list[StepLabel]:
    """Strict sensitivity labeler variant (does not replace default `label_trajectory`).

    Counts ONLY:
      - `exact_loop` repeated >= 3 times in a row (consecutive identical tool call + identical
        normalized observation; single repeats of streak length 2 and non-consecutive repeats are dropped)
      - `aba_thrash` lasting >= 3 cycles (>= 6 alternating tool steps A-B-A-B-A-B, or >= 5 steps
        A-B-A-B-A with all three A observations identical)
      - `period_k_rotation` (primitive period 2 <= k <= 5; for k=2 requires >= 3 cycles so 2-cycle
        ABAB is not re-admitted)
    Drops `near_duplicate`, single repeats, `redundant_reread`, and `error_streak`.
    """
    n = len(steps)
    results = [StepLabel(step_idx=s.step_idx) for s in steps]
    if n == 0:
        return results

    tool_positions = [i for i, s in enumerate(steps) if s.has_tool]
    m = len(tool_positions)
    if m == 0:
        return results

    # 1. Consecutive exact_loop streaks of length >= 3
    streak_start = 0
    while streak_start < m:
        pos0 = tool_positions[streak_start]
        s0 = steps[pos0]
        if not (s0.args_known and s0.obs_known):
            streak_start += 1
            continue
        streak_end = streak_start + 1
        while streak_end < m:
            pos_cur = tool_positions[streak_end]
            s_cur = steps[pos_cur]
            if (
                s_cur.args_known
                and s_cur.obs_known
                and s_cur.call_hash == s0.call_hash
                and s_cur.obs_norm_hash == s0.obs_norm_hash
            ):
                streak_end += 1
            else:
                break
        streak_len = streak_end - streak_start
        if streak_len >= 3:
            # Mark all repeat steps after the initial call in this >=3 consecutive streak
            for tp_i in range(streak_start + 1, streak_end):
                results[tool_positions[tp_i]].labels.add("exact_loop")
        streak_start = streak_end

    # 2. A-B-A thrash (>= 3 cycles) & Period-k rotation
    for tp_idx in range(m):
        pos_i = tool_positions[tp_idx]
        s_i = steps[pos_i]
        if not s_i.args_known:
            continue

        # Strict A-B-A thrash: >= 3 cycles (5 steps A-B-A-B-A with identical A obs, or 6 steps A-B-A-B-A-B)
        if tp_idx >= 4:
            p0, p1, p2, p3, p4 = (tool_positions[tp_idx - j] for j in (4, 3, 2, 1, 0))
            s0, s1, s2, s3, s4 = steps[p0], steps[p1], steps[p2], steps[p3], steps[p4]
            if (
                all(x.args_known for x in (s0, s1, s2, s3, s4))
                and s0.call_hash == s2.call_hash == s4.call_hash
                and s1.call_hash == s3.call_hash
                and s0.call_hash != s1.call_hash
            ):
                same_a_obs = (
                    s0.obs_known
                    and s2.obs_known
                    and s4.obs_known
                    and s0.obs_norm_hash == s2.obs_norm_hash == s4.obs_norm_hash
                )
                has_sixth = (
                    tp_idx >= 5
                    and steps[tool_positions[tp_idx - 5]].args_known
                    and steps[tool_positions[tp_idx - 5]].call_hash == s1.call_hash
                )
                if same_a_obs or has_sixth:
                    for p_mark in (p1, p2, p3, p4):
                        results[p_mark].labels.add("aba_thrash")

        # Period-k rotation (2 <= k <= 5; for k=2 require 3 full cycles = 6 steps)
        for k in range(2, 6):
            min_cycles = 3 if k == 2 else 2
            if tp_idx < min_cycles * k - 1:
                continue
            block2_pos = tool_positions[tp_idx - k + 1 : tp_idx + 1]
            block1_pos = tool_positions[tp_idx - 2 * k + 1 : tp_idx - k + 1]
            if not all(steps[p].args_known for p in block1_pos + block2_pos):
                continue
            hashes2 = [steps[p].call_hash for p in block2_pos]
            hashes1 = [steps[p].call_hash for p in block1_pos]
            if hashes1 != hashes2 or len(set(hashes2)) < 2:
                continue
            is_primitive = True
            for sub_k in range(1, k):
                if k % sub_k == 0 and hashes2 == hashes2[:sub_k] * (k // sub_k):
                    is_primitive = False
                    break
            if not is_primitive:
                continue

            has_three_cycles = False
            if tp_idx >= 3 * k - 1:
                block0_pos = tool_positions[tp_idx - 3 * k + 1 : tp_idx - 2 * k + 1]
                if all(steps[p].args_known for p in block0_pos) and [
                    steps[p].call_hash for p in block0_pos
                ] == hashes2:
                    has_three_cycles = True

            if k == 2 and not has_three_cycles:
                continue

            any_same_obs = any(
                steps[p2].obs_known
                and steps[p1].obs_known
                and steps[p2].obs_norm_hash == steps[p1].obs_norm_hash
                for p1, p2 in zip(block1_pos, block2_pos)
            )
            has_write = any(
                steps[p].file_op in ("write", "write_any") for p in block1_pos + block2_pos
            )
            if any_same_obs or not has_write or has_three_cycles:
                for p in block2_pos:
                    results[p].labels.add("period_k_rotation")
                    if results[p].rotation_k is None or k < results[p].rotation_k:
                        results[p].rotation_k = k
                break

    for lbl in results:
        if lbl.labels & STRICT_LOOP_LABELS:
            lbl.is_loop = True
            for cand in _STRICT_PRIORITY:
                if cand in lbl.labels:
                    lbl.primary_label = cand
                    break
        else:
            lbl.is_loop = False
            lbl.primary_label = "normal"

    return results


_READONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read",
        "read_file",
        "view_file",
        "open_file",
        "open",
        "cat",
        "file_read",
        "get_file",
        "view",
        "grep",
        "glob",
        "search_file",
        "list_directory",
        "ls",
        "find_by_name",
        "grep_search",
    }
)

_SHELL_INSPECT_BINS: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "nl",
        "less",
        "more",
        "bat",
        "grep",
        "rg",
        "egrep",
        "fgrep",
        "find",
        "pwd",
        "wc",
        "stat",
        "file",
        "which",
        "whereis",
        "tree",
        "du",
        "df",
        "ps",
        "diff",
        "awk",
    }
)

_GIT_READONLY_SUBCMDS: frozenset[str] = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "rev-parse",
        "blame",
        "grep",
        "ls-files",
        "describe",
    }
)

_TEST_RUNNERS: frozenset[str] = frozenset(
    {
        "pytest",
        "py.test",
        "tox",
        "nox",
        "jest",
        "ctest",
    }
)

_MUTATION_SUBSTRINGS: tuple[str, ...] = (
    " > ",
    " >>",
    ">/",
    "tee ",
    "sed -i",
    "perl -i",
    "mv ",
    "cp ",
    "rm ",
    "mkdir ",
    "touch ",
    "chmod ",
    "chown ",
    "git checkout",
    "git reset",
    "git apply",
    "git commit",
    "git add",
    "git stash",
    "git clean",
    "patch ",
    "apply_patch",
    "pip install",
    "npm install",
    ".write(",
    "write_text(",
)


def _extract_cmd_text_from_canonical(canonical_call: str) -> str:
    """Extract inner command text from canonical tool call string like `bash({"command":"..."})`."""
    if "(" not in canonical_call or not canonical_call.endswith(")"):
        return canonical_call
    inner = canonical_call[canonical_call.find("(") + 1 : -1].strip()
    if inner.startswith("{") and inner.endswith("}"):
        m = re.search(r'"(?:command|cmd|terminal)"\s*:\s*"((?:[^"\\]|\\.)*)"', inner)
        if m:
            return m.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
    return inner


def classify_readonly_command(step: NormalizedStep) -> str | None:
    """Classify a step as a read-only command category or None if mutating/other.

    Categories returned:
      - 'read_or_search_tool': dedicated file-read or search tool (`Read`, `Grep`, `str_replace_editor view`, etc.)
      - 'shell_inspect': `ls`, `cat`, `head`, `tail`, `grep`, `rg`, `find`, `sed -n`, `wc`, `pwd`, etc.
      - 'git_readonly': `git status`, `git diff`, `git log`, `git show`, etc.
      - 'pytest_or_test_runner': `pytest`, `python -m pytest`, `unittest`, `cargo test`, `go test`, etc.
    """
    if not step.has_tool or not step.args_known:
        return None
    tname = step.tool_name.lower()
    if tname in _READONLY_TOOL_NAMES:
        return "read_or_search_tool"
    if step.file_op in ("write", "write_any"):
        return None
    if step.file_op == "read" and tname not in {"bash", "shell", "execute", "execute_bash", "terminal", "run_command", "exec_command", "cmd"}:
        return "read_or_search_tool"

    cmd_text = _extract_cmd_text_from_canonical(step.canonical_call)
    if not cmd_text:
        return None
    cmd_low = f" {cmd_text.lower()} "
    if any(mut in cmd_low for mut in _MUTATION_SUBSTRINGS):
        return None

    # Split compound `cd ... && <cmd>` or pipelines and inspect primary segments
    segments = [seg.strip() for seg in re.split(r"&&|;|\|", cmd_text) if seg.strip()]
    classified: list[str] = []
    for seg in segments:
        tokens = seg.split()
        # Strip leading env vars `FOO=bar` or wrapper `bash -lc` / `timeout 30`
        while tokens and ("=" in tokens[0] and not tokens[0].startswith("-")):
            tokens = tokens[1:]
        if len(tokens) >= 3 and tokens[0] in {"bash", "/bin/bash", "sh", "/bin/sh"} and tokens[1] in {"-c", "-lc"}:
            inner_seg = " ".join(tokens[2:]).strip("'\"")
            tokens = inner_seg.split()
        while tokens and tokens[0] in {"timeout", "time", "nice", "env", "nohup"}:
            tokens = tokens[1:]
            while tokens and (tokens[0].startswith("-") or tokens[0].isdigit()):
                tokens = tokens[1:]
        if not tokens:
            continue
        head = tokens[0].split("/")[-1].lower()
        if head in {"cd", "export", "set", "true", "echo", "printf"}:
            continue
        if head in _TEST_RUNNERS:
            classified.append("pytest_or_test_runner")
        elif head in {"python", "python3"} and (
            "-m pytest" in seg.lower()
            or "-m unittest" in seg.lower()
            or "pytest" in tokens[1:3]
            or "unittest" in tokens[1:3]
            or any(t.endswith("test.py") or t.startswith("test_") for t in tokens[1:])
        ):
            classified.append("pytest_or_test_runner")
        elif head in {"cargo", "go", "npm", "yarn", "pnpm", "mvn", "gradle"} and len(tokens) >= 2 and tokens[1].lower() == "test":
            classified.append("pytest_or_test_runner")
        elif head == "git" and len(tokens) >= 2 and tokens[1].lower() in _GIT_READONLY_SUBCMDS:
            classified.append("git_readonly")
        elif head in _SHELL_INSPECT_BINS:
            classified.append("shell_inspect")
        elif head == "sed" and "-n" in tokens and "-i" not in tokens:
            classified.append("shell_inspect")
        else:
            return None

    if not classified:
        return "shell_inspect" if step.file_op == "read" else None
    if "pytest_or_test_runner" in classified:
        return "pytest_or_test_runner"
    if "git_readonly" in classified:
        return "git_readonly"
    return classified[0]


def is_readonly_command_step(step: NormalizedStep) -> bool:
    """Return True if step is a read-only tool call or read-only shell/git/pytest command."""
    return classify_readonly_command(step) is not None

