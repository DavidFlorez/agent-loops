"""Canonicalization of tool calls, observations, error signatures, and file operations."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

import orjson

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_COMPACT_TS_RE = re.compile(r"\b\d{8}[T_]\d{6}\b")
_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
_EPOCH_RE = re.compile(r"\b1[5-9]\d{8}(?:\d{3})?\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TMP_DIR_RE = re.compile(r"(?:/var/folders|/var/tmp|/tmp|/dev/shm)/[^\s'\"`,;)}\]\\]+")
_TMP_TOKEN_RE = re.compile(r"\btmp[A-Za-z0-9_-]{5,}\b")
_PID_RE = re.compile(r"\b(?:pid|PID|pgid|PGID)[=:\s]+\d+\b")
_HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]{6,16}\b")
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|sec|secs|seconds|second|s)\b")
_WS_RE = re.compile(r"\s+")

# Next.js RSC flight reference pointers found in scraped Terminal-Bench traces (e.g. "$3b", "$4a")
RSC_HEX_REF_RE = re.compile(r"^\$[0-9a-fA-F]+$")

EMPTY_OBS_HASH = hashlib.sha256(b"__EMPTY_OBS__").hexdigest()


def is_rsc_ref(text: Any) -> bool:
    """Return True if text is an unresolved Next.js RSC hex chunk pointer like '$3b'."""
    if not isinstance(text, str):
        return False
    return bool(RSC_HEX_REF_RE.match(text.strip()))


def strip_ansi(text: str) -> str:
    """Remove ANSI terminal escape sequences."""
    return _ANSI_RE.sub("", text)


def normalize_text(
    text: str,
    *,
    strip_timestamps: bool = True,
    strip_temp_paths: bool = True,
    strip_runtime_noise: bool = False,
) -> str:
    """Normalize text by removing ANSI codes, timestamps, temp paths, and collapsing whitespace."""
    if not text:
        return ""
    out = _ANSI_RE.sub("", text) if "\x1b" in text else text
    if strip_timestamps and (":" in out or "-" in out or "_" in out or "1" in out):
        if "-" in out and ":" in out:
            out = _ISO_TS_RE.sub("<TS>", out)
        if "_" in out or "T" in out:
            out = _COMPACT_TS_RE.sub("<TS>", out)
        if ":" in out:
            out = _TIME_RE.sub("<TIME>", out)
        out = _EPOCH_RE.sub("<EPOCH>", out)
    if strip_temp_paths:
        if "-" in out:
            out = _UUID_RE.sub("<UUID>", out)
        if "/tmp" in out or "/var/" in out or "/dev/shm" in out:
            out = _TMP_DIR_RE.sub("<TMP_PATH>", out)
        if "tmp" in out:
            out = _TMP_TOKEN_RE.sub("<TMP_NAME>", out)
    if strip_runtime_noise:
        if "pid" in out or "PID" in out or "pgid" in out or "PGID" in out:
            out = _PID_RE.sub("pid=<PID>", out)
        if "0x" in out:
            out = _HEX_ADDR_RE.sub("<HEX>", out)
        out = _DURATION_RE.sub("<DUR>", out)
    return _WS_RE.sub(" ", out).strip()


def _normalize_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return normalize_text(obj)
    if isinstance(obj, dict):
        return {
            str(k).strip().lower(): _normalize_obj(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize_obj(v) for v in obj]
    return obj


def normalize_args(args: Any) -> str:
    """Canonicalize structured or string tool arguments into a normalized string."""
    if args is None:
        return ""
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = orjson.loads(stripped)
                norm = _normalize_obj(parsed)
                return json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                pass
        return normalize_text(stripped)
    norm = _normalize_obj(args)
    return json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonicalize_tool_call(tool_name: str, args: Any) -> str:
    """Return canonical representation of a tool call (normalized name + normalized args)."""
    name = (tool_name or "unknown").strip().lower()
    norm_args = normalize_args(args)
    return f"{name}({norm_args})"


def sha256_hex(text: str) -> str:
    """Return hex SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def hash_observation(obs: str | None, *, strip_noise: bool = False) -> str:
    """Hash a tool observation after ANSI/whitespace normalization (and optional runtime noise stripping)."""
    if obs is None:
        return EMPTY_OBS_HASH
    norm = normalize_text(
        obs,
        strip_timestamps=strip_noise,
        strip_temp_paths=strip_noise,
        strip_runtime_noise=strip_noise,
    )
    if not norm:
        return EMPTY_OBS_HASH
    return sha256_hex(norm)


def extract_call_tokens(canonical_call: str) -> tuple[str, ...]:
    """Tokenize a canonical call for fast near-duplicate similarity comparisons."""
    return tuple(re.findall(r"[A-Za-z0-9_./:-]+", canonical_call.lower()))


_EXIT_CODE_RE = re.compile(r"(?:exit\s*(?:code|status)\s*[:=]?\s*(-?\d+))", re.IGNORECASE)
_EXIT_CODE_XML_RE = re.compile(r"<exit_code>\s*(-?\d+)\s*</exit_code>", re.IGNORECASE)
_RETURNCODE_XML_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.IGNORECASE)
_EXECUTED_EXIT_RE = re.compile(r"executed with exit code\s*(-?\d+)", re.IGNORECASE)
_BENIGN_ERROR_LINE_RE = re.compile(r"^\s*error\s*:\s*\(?(?:none|null|0|false)\)?\s*$", re.IGNORECASE)
_ERROR_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:Traceback \(most recent call last\):|"
    r"[A-Za-z_]*(?:Error|Exception|Failure)\s*:.*|"
    r"FAILED\b.*|"
    r"ERROR\s*:\s*(?:\n.*|.+)|"
    r"fatal:.*|"
    r"error\[E\d+\]:.*|"
    r"command not found.*|"
    r"No such file or directory.*|"
    r"Permission denied.*|"
    r"\[error\] tool reported failure.*)",
    re.IGNORECASE,
)


def detect_error_signature(
    obs: str | None,
    *,
    status: str | None = None,
    exit_code: int | None = None,
    error_field: Any = None,
    msg: str | None = None,
) -> tuple[bool, str | None]:
    """Return (is_error, normalized_error_signature_hash)."""
    has_explicit_err = False
    if status and status.lower() in {"error", "failed", "timeout"}:
        has_explicit_err = True
    if exit_code is not None and exit_code != 0:
        has_explicit_err = True
    if error_field not in (None, "", False):
        has_explicit_err = True

    text = strip_ansi(obs or "")
    parsed_exit: int | None = exit_code
    for source_txt in (text, msg or ""):
        if parsed_exit is not None or not source_txt:
            continue
        for pat in (_EXIT_CODE_RE, _EXIT_CODE_XML_RE, _RETURNCODE_XML_RE, _EXECUTED_EXIT_RE):
            m = pat.search(source_txt)
            if m:
                try:
                    parsed_exit = int(m.group(1))
                    if parsed_exit != 0:
                        has_explicit_err = True
                    break
                except ValueError:
                    pass

    raw_matches = _ERROR_LINE_RE.findall(text) if text else []
    err_matches = [m for m in raw_matches if not _BENIGN_ERROR_LINE_RE.match(m.strip())]
    is_err = has_explicit_err or bool(err_matches)
    if not is_err:
        return False, None

    parts: list[str] = []
    if parsed_exit is not None and parsed_exit != 0:
        parts.append(f"exit={parsed_exit}")
    if error_field not in (None, "", False):
        parts.append(normalize_text(str(error_field), strip_runtime_noise=True))
    if err_matches:
        tail_lines = [normalize_text(line, strip_runtime_noise=True) for line in err_matches[-3:]]
        parts.extend(tail_lines)
    elif text:
        parts.append(normalize_text(text[-300:], strip_runtime_noise=True))
    elif msg:
        parts.append(normalize_text(msg[-200:], strip_runtime_noise=True))

    sig_text = "|".join(p for p in parts if p)
    if not sig_text:
        sig_text = "generic_error"
    return True, sha256_hex(sig_text)


_READ_TOOLS = {
    "read",
    "read_file",
    "view_file",
    "open_file",
    "open",
    "cat",
    "file_read",
    "get_file",
    "view",
}
_WRITE_TOOLS = {
    "write",
    "write_file",
    "edit",
    "edit_file",
    "multiedit",
    "multi_edit",
    "str_replace",
    "str_replace_based_edit_tool",
    "str_replace_editor",
    "apply_patch",
    "patch",
    "create_file",
    "create",
    "replace",
    "replace_in_file",
    "replace_file_content",
    "write_to_file",
    "insert",
    "delete",
}
_BASH_TOOLS = {
    "bash",
    "bash_command",
    "bash_terminal",
    "shell",
    "execute",
    "execute_bash",
    "terminal",
    "run_command",
    "run_shell_command",
    "exec_command",
    "cmd",
}
_ZERO_ARG_OK_TOOLS = {
    "finish",
    "submit",
    "mark_task_complete",
    "end_execution",
    "finish_verification",
}
# Tools in scraped Terminal-Bench traces whose `cmd` only records path/subcommand, omitting diff/offset
_COARSE_EDIT_TOOLS = {
    "edit",
    "write",
    "edit_file",
    "write_file",
    "replace",
    "str_replace_editor",
}

_SHELL_READ_CMDS = {"cat", "head", "tail", "nl", "less", "more", "bat"}
_SHELL_MUTATE_HINTS = (
    ">",
    ">>",
    "tee ",
    "sed -i",
    "mv ",
    "cp ",
    "rm ",
    "chmod ",
    "chown ",
    "mkdir ",
    "touch ",
    "git checkout",
    "git reset",
    "git apply",
    "patch ",
    "apply_patch",
)


def _extract_path_from_dict(d: dict[str, Any]) -> tuple[str | None, str | None]:
    path = None
    for k in (
        "file_path",
        "path",
        "target_file",
        "filename",
        "file",
        "AbsolutePath",
        "TargetFile",
    ):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            path = normalize_text(v, strip_timestamps=False, strip_temp_paths=False)
            break
    offset = d.get("offset", d.get("start_line", d.get("StartLine", d.get("view_range", d.get("line_start")))))
    limit = d.get("limit", d.get("end_line", d.get("EndLine", d.get("line_end"))))
    if offset is not None or limit is not None:
        return path, f"{offset}:{limit}"
    return path, "full"


def classify_file_op(tool_name: str, args: Any) -> tuple[str | None, str | None, str | None]:
    """Classify a tool call as ('read'|'write'|'write_any', normalized_path, range_key) or (None, None, None)."""
    name = (tool_name or "").strip().lower()
    parsed: dict[str, Any] | None = None
    if isinstance(args, dict):
        parsed = args
    elif isinstance(args, str) and args.strip().startswith("{"):
        try:
            obj = orjson.loads(args)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:
            parsed = None

    if name in _READ_TOOLS:
        if parsed:
            path, rkey = _extract_path_from_dict(parsed)
            return ("read", path, rkey) if path else (None, None, None)
        if isinstance(args, str) and args.strip() and not is_rsc_ref(args):
            return ("read", normalize_text(args, strip_timestamps=False, strip_temp_paths=False), "full")
        return (None, None, None)

    if name in _WRITE_TOOLS:
        if parsed:
            cmd_sub = str(parsed.get("command", "")).lower()
            path, rkey = _extract_path_from_dict(parsed)
            if cmd_sub == "view":
                return ("read", path, rkey) if path else (None, None, None)
            return ("write", path, None) if path else ("write_any", None, None)
        if isinstance(args, str) and args.strip() and not is_rsc_ref(args):
            arg_s = args.strip()
            if arg_s == "view":
                return (None, None, None)
            if arg_s in {"str_replace", "create", "insert"}:
                return ("write_any", None, None)
            return ("write", normalize_text(arg_s, strip_timestamps=False, strip_temp_paths=False), None)
        return ("write_any", None, None)

    if name in _BASH_TOOLS:
        cmd_str = ""
        if parsed:
            cmd_str = str(parsed.get("command") or parsed.get("cmd") or parsed.get("terminal") or "")
        elif isinstance(args, list):
            cmd_str = " ".join(str(x) for x in args if not is_rsc_ref(x))
        elif isinstance(args, str):
            cmd_str = args
        cmd_str = cmd_str.strip()
        if not cmd_str or is_rsc_ref(cmd_str):
            return (None, None, None)
        has_redirect = any(tok in cmd_str for tok in (">", ">>"))
        if has_redirect or any(hint in cmd_str for hint in _SHELL_MUTATE_HINTS):
            return ("write_any", None, None)
        if "|" not in cmd_str and "&&" not in cmd_str and ";" not in cmd_str:
            try:
                parts = shlex.split(cmd_str)
            except ValueError:
                parts = cmd_str.split()
            # Unwrap `bash -lc "..."` / `/bin/bash -lc "..."`
            if len(parts) >= 3 and parts[0] in {"bash", "/bin/bash", "sh", "/bin/sh"} and parts[1] in {"-c", "-lc"}:
                try:
                    parts = shlex.split(parts[2])
                except ValueError:
                    parts = parts[2].split()
            if parts and parts[0] in _SHELL_READ_CMDS:
                files = [p for p in parts[1:] if not p.startswith("-")]
                if len(files) == 1:
                    flags = " ".join(p for p in parts[1:] if p.startswith("-"))
                    return ("read", files[0], f"{parts[0]}:{flags or 'full'}")
            if parts and parts[0] == "sed" and "-n" in parts and "-i" not in parts:
                files = [p for p in parts[1:] if not p.startswith("-") and "p" not in p]
                if len(files) == 1:
                    range_expr = " ".join(p for p in parts[1:] if p != files[0])
                    return ("read", files[0], f"sed:{range_expr}")
        return (None, None, None)

    return (None, None, None)


@dataclass(slots=True)
class NormalizedStep:
    """Unified step representation across Terminal-Bench, SWE-rebench, and tau2-bench."""

    step_idx: int
    has_tool: bool
    tool_name: str
    canonical_call: str
    call_hash: str
    call_tokens: tuple[str, ...]
    args_known: bool
    obs_text: str
    obs_hash: str
    obs_norm_hash: str
    obs_known: bool
    is_error: bool
    error_sig: str | None
    file_op: str | None
    file_path: str | None
    file_range: str | None
    assistant_text: str = ""
    msg_chars: int = 0
    obs_chars: int = 0
    read_range_exact: bool = True


_OPENHANDS_ACTION_PREFIXES = (
    "running command:",
    "reading file:",
    "i am thinking",
    "running python code",
    "viewing ",
    "editing ",
    "writing ",
    "managing ",
    "clearing the task list",
)


def _is_openhands_action_announcement(step: dict[str, Any]) -> bool:
    obs = step.get("obs")
    if obs not in (None, ""):
        return False
    msg = (step.get("msg") or "").strip().lower()
    if not msg:
        return True
    return any(msg.startswith(pref) for pref in _OPENHANDS_ACTION_PREFIXES)


def collapse_openhands_tb_steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse OpenHands action-announcement + execution-result step pairs in Terminal-Bench."""
    out: list[dict[str, Any]] = []
    i = 0
    n = len(raw_steps)
    while i < n:
        cur = raw_steps[i]
        cur_tools = cur.get("tools") or []
        if (
            cur_tools
            and i + 1 < n
            and (raw_steps[i + 1].get("tools") or []) == cur_tools
            and _is_openhands_action_announcement(cur)
        ):
            nxt = raw_steps[i + 1]
            merged = dict(nxt)
            m0 = (cur.get("msg") or "").strip()
            m1 = (nxt.get("msg") or "").strip()
            merged["msg"] = f"{m0}\n{m1}".strip() if (m0 and m1) else (m0 or m1)
            out.append(merged)
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def make_step(
    step_idx: int,
    *,
    tool_name: str | None,
    raw_args: Any,
    obs: str | None,
    assistant_text: str = "",
    status: str | None = None,
    exit_code: int | None = None,
    error_field: Any = None,
    is_tb_coarse_schema: bool = False,
    msg_chars_override: int | None = None,
    obs_chars_override: int | None = None,
) -> NormalizedStep:
    """Build a NormalizedStep from raw tool call and observation fields."""
    if not tool_name:
        return NormalizedStep(
            step_idx=step_idx,
            has_tool=False,
            tool_name="",
            canonical_call="",
            call_hash="",
            call_tokens=(),
            args_known=False,
            obs_text=obs or "",
            obs_hash=EMPTY_OBS_HASH,
            obs_norm_hash=EMPTY_OBS_HASH,
            obs_known=False,
            is_error=False,
            error_sig=None,
            file_op=None,
            file_path=None,
            file_range=None,
            assistant_text=assistant_text,
            msg_chars=msg_chars_override if msg_chars_override is not None else len(assistant_text),
            obs_chars=obs_chars_override if obs_chars_override is not None else len(obs or ""),
        )

    tname = tool_name.strip().lower()
    # Check whether arguments are genuinely resolved and complete
    args_known = True
    if is_rsc_ref(raw_args):
        args_known = False
    elif isinstance(raw_args, list) and any(is_rsc_ref(x) for x in raw_args):
        # Note: Codex shell `["apply_patch", "$33"]` has unresolved patch payload
        args_known = False
    elif (raw_args is None or (isinstance(raw_args, str) and not raw_args.strip())) and tname not in _ZERO_ARG_OK_TOOLS:
        args_known = False
    elif is_tb_coarse_schema and tname in _COARSE_EDIT_TOOLS and isinstance(raw_args, str):
        # Terminal-Bench Edit/Write/replace/str_replace_editor only stores path or verb, omitting diff
        args_known = False

    if args_known:
        canonical = canonicalize_tool_call(tname, raw_args)
    else:
        canonical = f"{tname}(__UNRESOLVED_{step_idx}__)"

    call_hash = sha256_hex(canonical)
    tokens = extract_call_tokens(canonical) if args_known else ()

    obs_str = obs if isinstance(obs, str) else ("" if obs is None else str(obs))
    obs_known = not is_rsc_ref(obs_str)
    if obs_known:
        obs_nh = hash_observation(obs_str, strip_noise=True)
        obs_h = obs_nh
    else:
        obs_h = sha256_hex(f"__UNRESOLVED_OBS_{step_idx}__")
        obs_nh = obs_h

    _pure_read_or_search = _READ_TOOLS | {"grep", "glob", "search_file", "list_directory", "ls"}
    if tname in _pure_read_or_search and (status or "").lower() != "error" and exit_code in (None, 0) and not error_field:
        is_err, err_sig = False, None
    else:
        is_err, err_sig = detect_error_signature(
            obs_str if obs_known else None,
            status=status,
            exit_code=exit_code,
            error_field=error_field,
            msg=assistant_text,
        )
    fop, fpath, frange = classify_file_op(tname, raw_args)
    read_exact = not (is_tb_coarse_schema and tname in {"read", "read_file"})

    msg_chars = (
        msg_chars_override
        if msg_chars_override is not None
        else len(assistant_text) + len(tool_name) + len(str(raw_args or ""))
    )
    obs_chars = obs_chars_override if obs_chars_override is not None else len(obs_str)

    return NormalizedStep(
        step_idx=step_idx,
        has_tool=True,
        tool_name=tname,
        canonical_call=canonical,
        call_hash=call_hash,
        call_tokens=tokens,
        args_known=args_known,
        obs_text=obs_str if obs_known else "",
        obs_hash=obs_h,
        obs_norm_hash=obs_nh,
        obs_known=obs_known,
        is_error=is_err,
        error_sig=err_sig,
        file_op=fop,
        file_path=fpath,
        file_range=frange,
        assistant_text=assistant_text,
        msg_chars=msg_chars,
        obs_chars=obs_chars,
        read_range_exact=read_exact,
    )
