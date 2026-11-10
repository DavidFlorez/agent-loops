"""Download and locally cache all four public trajectory datasets, then inspect schemas."""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pyarrow.parquet as pq
import requests
from huggingface_hub import snapshot_download

TB_REPO = "yoonholee/terminalbench-trajectories"
SWE_REPO = "ibragim-bad/swe_rebench_07_2026_trajectories"
LONGTAIL_REPO = "nebius/Longtail-SWE-trajectories"
SIERRA_S3_URL = "https://sierra-tau-bench-public.s3.amazonaws.com"
TAU2_CONTENTS_API = (
    "https://api.github.com/repos/sierra-research/tau2-bench/contents/data/tau2/results/final"
)

DEFAULT_DATA_DIR = Path("data")

_VOICE_TOKENS = ("voice", "nova", "gpt-realtime", "native-audio")


def download_terminalbench(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download Terminal-Bench 2.0 trajectories parquet files if not cached."""
    target = data_dir / "terminalbench"
    parquets = sorted(target.glob("data/*.parquet"))
    if len(parquets) >= 2:
        print(f"[cache hit] Terminal-Bench already cached at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {TB_REPO} -> {target}")
    snapshot_download(
        repo_id=TB_REPO,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["README.md", "data/*.parquet"],
    )
    return target


def download_swe_rebench(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download SWE-rebench July 2026 trajectories if not cached."""
    target = data_dir / "swe_rebench_07_2026"
    shards = sorted(target.glob("trajectories/*/*.jsonl.gz"))
    if len(shards) >= 85:
        print(f"[cache hit] SWE-rebench July 2026 already cached at {target} ({len(shards)} shards)")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {SWE_REPO} -> {target}")
    snapshot_download(
        repo_id=SWE_REPO,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=[
            "README.md",
            "schema.json",
            "manifest.json",
            "index.jsonl",
            "trajectories/**/*.jsonl.gz",
        ],
    )
    return target


def download_longtail_swe(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download Longtail-SWE trajectories and manifests if not cached."""
    target = data_dir / "longtail_swe"
    parquets = sorted((target / "data").glob("*.parquet"))
    if len(parquets) >= 9 and (target / "README.md").exists():
        print(f"[cache hit] Longtail-SWE already cached at {target} ({len(parquets)} parquet shards)")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {LONGTAIL_REPO} -> {target}")
    snapshot_download(
        repo_id=LONGTAIL_REPO,
        repo_type="dataset",
        local_dir=str(target),
    )
    return target


def _list_s3_keys(prefix: str) -> list[str]:
    """List keys under the public Sierra tau2 S3 bucket prefix."""
    keys: list[str] = []
    token: str | None = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        resp = requests.get(SIERRA_S3_URL, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for c in root.findall("s3:Contents", ns):
            k = c.findtext("s3:Key", default="", namespaces=ns)
            if k:
                keys.append(k)
        is_trunc = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        if is_trunc.lower() == "true":
            token = root.findtext("s3:NextContinuationToken", default="", namespaces=ns)
            if not token:
                break
        else:
            break
    return keys


def download_sierra_tau2_public(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Cache non-voice _sierra_ submissions from s3://sierra-tau-bench-public/submissions/."""
    target = data_dir / "sierra_tau2_public"
    existing = sorted(p for p in target.iterdir() if p.is_dir()) if target.exists() else []
    if len(existing) >= 27 and all((p / "submission.json").exists() for p in existing):
        print(
            f"[cache hit] Sierra tau2 public S3 already cached at {target} "
            f"({len(existing)} _sierra_ submission folders)"
        )
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"[listing] {SIERRA_S3_URL}/submissions/ (_sierra_ text/tool submissions)")
    all_keys = _list_s3_keys("submissions/")
    by_sub: dict[str, list[str]] = defaultdict(list)
    for k in all_keys:
        parts = k.split("/", 2)
        if len(parts) < 3:
            continue
        sub_name = parts[1]
        if "_sierra_" not in sub_name:
            continue
        low = sub_name.lower()
        if any(vt in low for vt in _VOICE_TOKENS):
            continue
        if k.endswith(".wav") or "/audio/" in k:
            continue
        by_sub[sub_name].append(k)

    for sub_name, keys in sorted(by_sub.items()):
        has_sim_dirs = any("/simulations/" in k for k in keys)
        for k in keys:
            rel = k.split(f"submissions/{sub_name}/", 1)[1]
            # Skip duplicate domain-level monolith JSONs if per-simulation files exist
            if has_sim_dirs and rel.startswith("trajectories/") and rel.count("/") == 1 and rel.endswith(".json"):
                continue
            dest = target / sub_name / rel
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{SIERRA_S3_URL}/{k}"
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with dest.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
    return target


def download_tau2_results(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download tau2-bench bundled final result JSON files if not cached."""
    target = data_dir / "tau2_results"
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "_files.json"

    if manifest_path.exists():
        files = json.loads(manifest_path.read_text())
    else:
        resp = requests.get(TAU2_CONTENTS_API, timeout=60)
        resp.raise_for_status()
        entries = resp.json()
        files = [
            {"name": e["name"], "download_url": e["download_url"], "size": e["size"]}
            for e in entries
            if e.get("type") == "file" and e["name"].endswith(".json")
        ]
        manifest_path.write_text(json.dumps(files, indent=2))

    cached_count = 0
    for f in files:
        dest = target / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            cached_count += 1
            continue
        print(f"[downloading] tau2 {f['name']} ({f['size'] / 1e6:.1f} MB)")
        with requests.get(f["download_url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with dest.open("wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
    if cached_count == len(files):
        print(f"[cache hit] tau2-bench results already cached at {target} ({cached_count} files)")
    return target


def inspect_schemas(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Inspect and print schemas plus token/cost/step field coverage across all cached datasets."""
    info: dict[str, Any] = {}

    # 1. Terminal-Bench 2.0
    tb_dir = data_dir / "terminalbench"
    parquets = sorted(tb_dir.glob("data/*.parquet"))
    if parquets:
        pf = pq.ParquetFile(parquets[0])
        schema_str = str(pf.schema_arrow)
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
        print("=" * 72)
        print(f"SCHEMA 1: {TB_REPO} ({len(parquets)} parquet shards, {total_rows:,} rows)")
        print("=" * 72)
        print(schema_str)
        sample_step_keys: list[str] = []
        sample_tool_keys: list[str] = []
        for batch in pf.iter_batches(batch_size=500, columns=["steps"]):
            for raw_st in batch.column("steps").to_pylist():
                if not raw_st or raw_st == "null":
                    continue
                steps = json.loads(raw_st)
                if steps:
                    sample_step_keys = sorted({k for s in steps for k in s.keys()})
                    for s in steps:
                        if s.get("tools"):
                            sample_tool_keys = sorted({k for t in s["tools"] for k in t.keys()})
                            break
                    if sample_tool_keys:
                        break
            if sample_tool_keys:
                break
        print("Sample step object keys:", sample_step_keys)
        print("Sample tool object keys:", sample_tool_keys)

        tbl = pq.read_table(
            parquets,
            columns=["agent", "model", "task_name", "input_tokens", "output_tokens", "cache_tokens", "cost_cents", "steps"],
        )
        slice_cov: list[dict[str, Any]] = []
        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in tbl.to_pylist():
            ag = str(r["agent"])
            if ag not in {"terminus-2", "openhands", "mini-swe-agent"}:
                continue
            key = (ag, str(r["model"]))
            g = grouped[key]
            g["rows"] += 1
            if r["steps"] and r["steps"] != "null" and len(r["steps"]) > 2:
                g["has_steps"] += 1
            if (r["input_tokens"] or 0) > 0 or (r["output_tokens"] or 0) > 0:
                g["has_tokens"] += 1
            if (r["cache_tokens"] or 0) > 0:
                g["has_cache"] += 1
            if (r["cost_cents"] or 0) > 0:
                g["has_cost"] += 1
        print("Token/Cost coverage in shared TB harnesses (terminus-2, openhands, mini-swe-agent):")
        for (ag, mod), g in sorted(grouped.items()):
            print(
                f"  {ag:15s} | {mod:52s} | n={g['rows']:4d} steps={g['has_steps']:4d} "
                f"tok={g['has_tokens']:4d} cache={g['has_cache']:4d} cost>0={g['has_cost']:4d}"
            )
            slice_cov.append({"agent": ag, "model": mod, **g})

        info["terminalbench"] = {
            "shards": len(parquets),
            "rows": total_rows,
            "columns": [f.name for f in pf.schema_arrow],
            "step_keys": sample_step_keys,
            "tool_keys": sample_tool_keys,
            "target_slice_models": len(slice_cov),
        }

    # 2. SWE-rebench July 2026
    swe_dir = data_dir / "swe_rebench_07_2026"
    schema_path = swe_dir / "schema.json"
    if schema_path.exists():
        schema_json = json.loads(schema_path.read_text())
        manifest_json = json.loads((swe_dir / "manifest.json").read_text())
        print("=" * 72)
        print(
            f"SCHEMA 2: {SWE_REPO} ({manifest_json.get('rows')} rows, "
            f"{manifest_json.get('shards')} shards, {manifest_json.get('participants')} participants)"
        )
        print("=" * 72)
        print("Top-level properties:", list(schema_json["properties"].keys()))
        print(
            "Event types:",
            schema_json["properties"]["events"]["items"]["properties"]["type"]["enum"],
        )
        swe_cov: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
        for shard in sorted(swe_dir.glob("trajectories/*/*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    part = rec.get("participant", {})
                    if part.get("type") != "common_scaffold_model":
                        continue
                    pk = str(part.get("key"))
                    g = swe_cov[pk]
                    g["model"] = str(part.get("model"))
                    g["rows"] += 1
                    u = rec.get("usage") or {}
                    tok = u.get("tokens") or {}
                    if (tok.get("total") or 0) > 0:
                        g["has_tokens"] += 1
                    if (tok.get("cached_input") or 0) > 0:
                        g["has_cache"] += 1
                    if (u.get("cost_usd") or 0) > 0:
                        g["has_cost"] += 1
        print("Token/Cost coverage in SWE-rebench common_scaffold_model (swelike):")
        for pk, g in sorted(swe_cov.items()):
            print(
                f"  {pk:20s} ({g['model']:26s}) | n={g['rows']:3d} tok={g['has_tokens']:3d} "
                f"cache={g['has_cache']:3d} cost>0={g['has_cost']:3d}"
            )
        info["swe_rebench"] = {
            "rows": manifest_json.get("rows"),
            "shards": manifest_json.get("shards"),
            "participants": manifest_json.get("participants"),
            "common_scaffold_models": len(swe_cov),
        }

    # 3. Longtail-SWE
    lt_dir = data_dir / "longtail_swe"
    lt_parquets = sorted((lt_dir / "data").glob("*.parquet"))
    if lt_parquets:
        lt_pf0 = pq.ParquetFile(lt_parquets[0])
        lt_total = sum(pq.ParquetFile(p).metadata.num_rows for p in lt_parquets)
        print("=" * 72)
        print(f"SCHEMA 3: {LONGTAIL_REPO} ({len(lt_parquets)} parquet shards, {lt_total:,} rows)")
        print("=" * 72)
        print(lt_pf0.schema_arrow)
        sample_row = lt_pf0.read_row_group(0).to_pylist()[0]
        sample_traj = json.loads(sample_row["trajectory"])
        print("Nested trajectory keys:", list(sample_traj.keys()))
        print("Longtail-SWE model coverage (swelike):")
        for p in lt_parquets:
            rows = pq.read_table(p).to_pylist()
            n = len(rows)
            graded = sum(1 for r in rows if r["resolved"] is not None)
            tok = cache = cost = zero_steps = 0
            ctxs = sorted({int(r["effective_context_tokens"]) for r in rows if r["effective_context_tokens"]})
            for r in rows:
                tj = json.loads(r["trajectory"])
                u = tj.get("usage") or {}
                tk = u.get("tokens") or {}
                if (u.get("steps") or 0) == 0:
                    zero_steps += 1
                if (tk.get("total") or 0) > 0:
                    tok += 1
                if (tk.get("cached_input") or 0) > 0:
                    cache += 1
                if (u.get("cost_usd") or 0) > 0:
                    cost += 1
            print(
                f"  {rows[0]['agent_config_id']:30s} | n={n:3d} graded={graded:3d} "
                f"zero_step={zero_steps:2d} ctx={ctxs} tok={tok:3d} cache={cache:3d} cost>0={cost:3d}"
            )
        info["longtail_swe"] = {
            "shards": len(lt_parquets),
            "rows": lt_total,
            "columns": [f.name for f in lt_pf0.schema_arrow],
            "trajectory_keys": list(sample_traj.keys()),
        }

    # 4. Sierra tau2 public S3
    sierra_dir = data_dir / "sierra_tau2_public"
    sierra_subs = sorted(p for p in sierra_dir.iterdir() if p.is_dir()) if sierra_dir.exists() else []
    if sierra_subs:
        print("=" * 72)
        print(f"SCHEMA 4: s3://sierra-tau-bench-public ({len(sierra_subs)} non-voice _sierra_ folders)")
        print("=" * 72)
        sample_bk = next(
            (p / "trajectories" / "banking_knowledge" / "results.json" for p in sierra_subs if (p / "trajectories" / "banking_knowledge" / "results.json").exists()),
            None,
        )
        if sample_bk:
            bk_obj = json.loads(sample_bk.read_text())
            print("results.json top keys:", list(bk_obj.keys()))
            print("results.json info keys:", list((bk_obj.get("info") or {}).keys()))
            sim_sample_p = next((sample_bk.parent / "simulations").glob("*.json"))
            sim_obj = json.loads(sim_sample_p.read_text())
            print("simulation JSON keys:", list(sim_obj.keys()))
            print(
                "message keys union (sample):",
                sorted({k for m in sim_obj.get("messages", []) for k in m.keys()}),
            )

        print("Sierra submission inventory (cohort | retrieval_config | user_sim | sims | tok | cost>0):")
        for sub in sierra_subs:
            tdir = sub / "trajectories"
            sub_dirs = sorted(d for d in tdir.iterdir() if d.is_dir()) if tdir.exists() else []
            if sub_dirs:
                for d in sub_dirs:
                    rf = d / "results.json"
                    info_j = json.loads(rf.read_text()).get("info", {}) if rf.exists() else {}
                    rcfg = info_j.get("retrieval_config")
                    usim = (info_j.get("user_info") or {}).get("llm")
                    sim_files = list((d / "simulations").glob("*.json"))
                    tok_c = cost_c = 0
                    for sp in sim_files:
                        s = json.loads(sp.read_text())
                        if (s.get("agent_cost") or 0) > 0:
                            cost_c += 1
                        if any(
                            isinstance(m.get("usage"), dict)
                            and ((m["usage"].get("prompt_tokens") or 0) > 0 or (m["usage"].get("completion_tokens") or 0) > 0)
                            for m in s.get("messages", [])
                            if m.get("role") == "assistant"
                        ):
                            tok_c += 1
                    print(
                        f"  {sub.name:42s} | {d.name:18s} | rcfg={str(rcfg):16s} | "
                        f"user={str(usim):10s} | n={len(sim_files):3d} tok={tok_c:3d} cost>0={cost_c:3d}"
                    )
            else:
                for mf in sorted(tdir.glob("*.json")):
                    obj = json.loads(mf.read_text())
                    info_j = obj.get("info") or {}
                    sims = obj.get("simulations") or []
                    usim = (info_j.get("user_info") or {}).get("llm")
                    dom = (info_j.get("environment_info") or {}).get("domain_name") or mf.stem
                    print(f"  {sub.name:42s} | {str(dom):18s} | monolith | user={usim} | n={len(sims):3d}")
        info["sierra_tau2_public"] = {"submissions": len(sierra_subs)}

    # Legacy GitHub tau2_results sanity check files
    tau2_dir = data_dir / "tau2_results"
    tau2_files = sorted(p for p in tau2_dir.glob("*.json") if p.name != "_files.json") if tau2_dir.exists() else []
    if tau2_files:
        info["tau2_github_final"] = {"files": len(tau2_files)}

    return info


def download_all(data_dir: Path = DEFAULT_DATA_DIR, include_tau2: bool = True) -> dict[str, Any]:
    """Download/verify cache for all datasets and inspect their schemas."""
    download_terminalbench(data_dir)
    download_swe_rebench(data_dir)
    download_longtail_swe(data_dir)
    if include_tau2:
        download_sierra_tau2_public(data_dir)
        download_tau2_results(data_dir)
    return inspect_schemas(data_dir)


if __name__ == "__main__":
    download_all()
