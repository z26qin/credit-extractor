"""CUAD loading for the Milestone 0 reproduction (official 408/102 split)."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from .chunking import Chunk, chunk_text
from .prompts import REPO_ROOT, canon_map

CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def download_cuad(dest_dir: Path) -> Path:
    """Fetch the official CUAD data.zip and extract CUADv1.json (cached)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "CUADv1.json"
    if out.exists():
        return out
    zpath = dest_dir / "data.zip"
    urllib.request.urlretrieve(CUAD_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("CUADv1.json", dest_dir)
    return out


def load_contracts(cuad_json: Path) -> dict[str, dict]:
    data = json.loads(Path(cuad_json).read_text(encoding="utf-8"))["data"]
    return {c["title"]: c for c in data}


def test_titles(manifest: Path | None = None) -> list[str]:
    manifest = manifest or REPO_ROOT / "data" / "split_manifest.json"
    return list(json.loads(Path(manifest).read_text(encoding="utf-8"))["test"])


def sample_titles(titles: list[str], n: int) -> list[str]:
    """Deterministic sample: first n by md5(title) (the upstream data-curve rule)."""
    return sorted(titles, key=lambda t: hashlib.md5(t.encode()).hexdigest())[:n]


def gold_for(contract: dict, clauses: list[str]) -> dict[str, list[str]]:
    """All answer spans per clause for one contract (contract-level gold)."""
    canon = canon_map(clauses)
    gold = {c: [] for c in clauses}
    for q in contract["paragraphs"][0]["qas"]:
        raw = q["id"].split("__")[-1]
        cl = canon.get("".join(ch for ch in raw.lower() if ch.isalnum()))
        if cl is None:
            raise ValueError(f"unmapped CUAD clause in qas id: {raw!r}")
        for a in q.get("answers", []):
            if a["text"] not in gold[cl]:
                gold[cl].append(a["text"])
    return gold


def contract_text(contract: dict) -> str:
    return contract["paragraphs"][0]["context"]


def build_chunks(contracts: dict[str, dict], titles: list[str],
                 size: int, overlap: int) -> list[Chunk]:
    out = []
    for t in titles:
        out.extend(chunk_text(t, contract_text(contracts[t]), size, overlap))
    return out
