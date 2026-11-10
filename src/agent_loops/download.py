"""Download and locally cache public trajectory datasets, then print schemas."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from huggingface_hub import snapshot_download

TB_REPO = "yoonholee/terminalbench-trajectories"
SWE_REPO = "ibragim-bad/swe_rebench_07_2026_trajectories"
TAU2_CONTENTS_API = (
    "https://api.github.com/repos/sierra-research/tau2-bench/contents/data/tau2/results/final"
)

DEFAULT_DATA_DIR = Path("data")


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
    """Inspect and print schemas for all cached datasets."""
    info: dict[str, Any] = {}

    tb_dir = data_dir / "terminalbench"
    parquets = sorted(tb_dir.glob("data/*.parquet"))
    if parquets:
        pf = pq.ParquetFile(parquets[0])
        schema_str = str(pf.schema_arrow)
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquets)
        print("=" * 72)
        print(f"SCHEMA: {TB_REPO} ({len(parquets)} parquet shards, {total_rows:,} rows)")
        print("=" * 72)
        print(schema_str)
        # Inspect first non-empty steps row
        sample_step_keys: list[str] = []
        sample_tool_keys: list[str] = []
        for batch in pf.iter_batches(batch_size=500, columns=["steps"]):
            for raw_st in batch.column("steps").to_pylist():
                if not raw_st:
                    continue
                steps = json.loads(raw_st)
                if steps:
                    sample_step_keys = sorted({k for s in steps for k in s.keys()})
                    for s in steps:
                        if s.get("tools"):
                            sample_tool_keys = sorted(
                                {k for t in s["tools"] for k in t.keys()}
                            )
                            break
                    if sample_tool_keys:
                        break
            if sample_tool_keys:
                break
        print("Sample step object keys:", sample_step_keys)
        print("Sample tool object keys:", sample_tool_keys)
        info["terminalbench"] = {
            "shards": len(parquets),
            "rows": total_rows,
            "columns": [f.name for f in pf.schema_arrow],
            "step_keys": sample_step_keys,
            "tool_keys": sample_tool_keys,
        }

    swe_dir = data_dir / "swe_rebench_07_2026"
    schema_path = swe_dir / "schema.json"
    if schema_path.exists():
        schema_json = json.loads(schema_path.read_text())
        manifest_json = json.loads((swe_dir / "manifest.json").read_text())
        print("=" * 72)
        print(
            f"SCHEMA: {SWE_REPO} ({manifest_json.get('rows')} rows, "
            f"{manifest_json.get('shards')} shards, {manifest_json.get('participants')} participants)"
        )
        print("=" * 72)
        print("Top-level properties:", list(schema_json["properties"].keys()))
        print(
            "Event types:",
            schema_json["properties"]["events"]["items"]["properties"]["type"]["enum"],
        )
        # Inspect usage fields from one row per participant shard (run_0)
        usage_keys_by_participant: dict[str, list[str]] = {}
        for shard in sorted(swe_dir.glob("trajectories/*/run_0.jsonl.gz")):
            participant = shard.parent.name
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                first_line = fh.readline()
                if first_line:
                    rec = json.loads(first_line)
                    u = rec.get("usage", {})
                    tok = u.get("tokens", {}) if isinstance(u.get("tokens"), dict) else {}
                    usage_keys_by_participant[participant] = {
                        "usage_keys": sorted(u.keys()),
                        "token_keys": sorted(tok.keys()),
                        "cost_usd": u.get("cost_usd"),
                    }
        print("Sample usage keys across participants (run_0 row 0):")
        for p, uk in usage_keys_by_participant.items():
            print(f"  {p:20s}: usage={uk['usage_keys']} tokens={uk['token_keys']}")
        info["swe_rebench"] = {
            "rows": manifest_json.get("rows"),
            "shards": manifest_json.get("shards"),
            "participants": manifest_json.get("participants"),
            "top_level_properties": list(schema_json["properties"].keys()),
        }

    tau2_dir = data_dir / "tau2_results"
    tau2_files = sorted(p for p in tau2_dir.glob("*.json") if p.name != "_files.json")
    if tau2_files:
        print("=" * 72)
        print(f"SCHEMA: sierra-research/tau2-bench ({len(tau2_files)} result JSON files)")
        print("=" * 72)
        sample_obj = json.loads(tau2_files[0].read_text())
        print("Top-level keys:", list(sample_obj.keys()))
        sims = sample_obj.get("simulations", [])
        if sims:
            print("Sample simulation keys:", list(sims[0].keys()))
            msgs = sims[0].get("messages", [])
            if msgs:
                print("Sample message keys:", sorted({k for m in msgs for k in m.keys()}))
        info["tau2"] = {
            "files": len(tau2_files),
            "top_level_keys": list(sample_obj.keys()),
        }

    return info


def download_all(data_dir: Path = DEFAULT_DATA_DIR, include_tau2: bool = True) -> dict[str, Any]:
    """Download all datasets and inspect their schemas."""
    download_terminalbench(data_dir)
    download_swe_rebench(data_dir)
    if include_tau2:
        download_tau2_results(data_dir)
    return inspect_schemas(data_dir)


if __name__ == "__main__":
    download_all()
