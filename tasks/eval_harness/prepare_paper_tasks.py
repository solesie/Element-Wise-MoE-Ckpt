#!/usr/bin/env python3
"""Download and materialize the fixed zero-shot paper evaluation tasks.

The prepared JSONL files deliberately remove the runtime dependency on the old
``datasets==1.15.1`` expected by lm-eval 0.2.0.  Raw sources and revisions are
fixed below and every output is checksummed in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable
from urllib.request import urlopen


TASK_DEFINITION_COMMIT = "f4d4b3de3ee6741a7151a9fe74945ee515262f4c"
DEFAULT_DATA_ROOT = Path.home() / "data" / "lm_eval_paper_6tasks"
PAPER_TASKS = (
    "piqa",
    "hellaswag",
    "race",
    "mathqa",
    "boolq",
    "arc_challenge",
    "winogrande",
)

HF_SOURCES = {
    "piqa": {
        "repo": "baber/piqa",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "filename": "piqa_validation.parquet",
        "raw_name": "piqa_validation.parquet",
        "sha256": "2b17ee1215b1ceb2ddabf4c90a05c5a8b833bef9d207960f57d111c43a2080f3",
        "split": "validation",
    },
    "hellaswag": {
        "repo": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "filename": "data/validation-00000-of-00001.parquet",
        "raw_name": "validation-00000-of-00001.parquet",
        "sha256": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        "split": "validation",
    },
    "race": {
        "repo": "EleutherAI/race",
        "revision": "66ca2e6374282f5e35b11559e87185db911a6900",
        "filename": "high/test-00000-of-00001.parquet",
        "raw_name": "test-00000-of-00001.parquet",
        "sha256": "b07da27f803881229a3208563c6b034b12b3ab5fc6f51140ad9a3825ee6949ef",
        "split": "high/test",
    },
    "boolq": {
        "repo": "aps/super_glue",
        "revision": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
        "filename": "boolq/validation-00000-of-00001.parquet",
        "raw_name": "validation-00000-of-00001.parquet",
        "sha256": "f46438b8cce048184552a2bfcdb8252eadb3a2602f0c22334caccd9b6ea547d6",
        "split": "validation",
    },
    "arc_challenge": {
        "repo": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "filename": "ARC-Challenge/test-00000-of-00001.parquet",
        "raw_name": "test-00000-of-00001.parquet",
        "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "split": "test",
    },
    "winogrande": {
        "repo": "allenai/winogrande",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "filename": "winogrande_xl/validation-00000-of-00001.parquet",
        "raw_name": "validation-00000-of-00001.parquet",
        "sha256": "f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1",
        "split": "validation",
    },
}

MATHQA_SOURCE = {
    "url": "https://math-qa.github.io/math-QA/data/MathQA.zip",
    "raw_name": "MathQA.zip",
    "sha256": "7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a",
    "split": "test",
}

PRIMARY_METRICS = {
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "race": "acc",
    "mathqa": "acc_norm",
    "boolq": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(path: Path, expected: str) -> bool:
    return path.is_file() and _sha256(path) == expected


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_hf(task: str, spec: dict[str, str], raw_root: Path) -> Path:
    destination = raw_root / task / spec["raw_name"]
    if _valid(destination, spec["sha256"]):
        return destination

    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=spec["repo"],
            filename=spec["filename"],
            repo_type="dataset",
            revision=spec["revision"],
        )
    )
    if _sha256(cached) != spec["sha256"]:
        raise RuntimeError(f"source checksum mismatch for {task}: {cached}")
    _copy_atomic(cached, destination)
    return destination


def _download_mathqa(raw_root: Path) -> Path:
    destination = raw_root / "mathqa" / MATHQA_SOURCE["raw_name"]
    if _valid(destination, MATHQA_SOURCE["sha256"]):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urlopen(MATHQA_SOURCE["url"], timeout=180) as response:
            shutil.copyfileobj(response, tmp, length=1024 * 1024)
    try:
        if _sha256(tmp_path) != MATHQA_SOURCE["sha256"]:
            raise RuntimeError("source checksum mismatch for MathQA.zip")
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    return destination


def download_all(raw_root: Path, jobs: int, offline: bool) -> dict[str, Path]:
    expected = {
        task: raw_root / task / spec["raw_name"]
        for task, spec in HF_SOURCES.items()
        if task in PAPER_TASKS
    }
    expected["mathqa"] = raw_root / "mathqa" / MATHQA_SOURCE["raw_name"]
    if offline:
        bad = [
            task
            for task, path in expected.items()
            if not _valid(
                path,
                MATHQA_SOURCE["sha256"]
                if task == "mathqa"
                else HF_SOURCES[task]["sha256"],
            )
        ]
        if bad:
            raise RuntimeError(f"offline source verification failed: {', '.join(bad)}")
        return expected

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, 6))) as pool:
        for task, spec in HF_SOURCES.items():
            if task not in PAPER_TASKS:
                continue
            futures[pool.submit(_download_hf, task, spec, raw_root)] = task
        futures[pool.submit(_download_mathqa, raw_root)] = "mathqa"
        for future in as_completed(futures):
            task = futures[future]
            path = future.result()
            print(f"downloaded {task}: {path} ({path.stat().st_size / 2**20:.2f} MiB)")
    return expected


def _preprocess_hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def _piqa(path: Path) -> Iterable[dict]:
    for row in _read_parquet(path):
        yield {
            "goal": row["goal"],
            "choices": [row["sol1"], row["sol2"]],
            "gold": int(row["label"]),
        }


def _hellaswag(path: Path) -> Iterable[dict]:
    for row in _read_parquet(path):
        context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        yield {
            "query": _preprocess_hellaswag(row["activity_label"] + ": " + context),
            "choices": [_preprocess_hellaswag(x) for x in row["endings"]],
            "gold": int(row["label"]),
        }


def _race(path: Path) -> Iterable[dict]:
    import ast

    for row in _read_parquet(path):
        problems = ast.literal_eval(row["problems"])
        if not problems:
            raise ValueError("RACE row contains no problems")
        yield {"article": row["article"], "problems": problems}


_MATHQA_CHOICES = re.compile(r"[abcd] \) .*?, |e \) .*?$")


def _mathqa(path: Path) -> Iterable[dict]:
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("test.json"))
    for row in rows:
        choices = [x[4:].rstrip(" ,") for x in _MATHQA_CHOICES.findall(row["options"])]
        if len(choices) != 5:
            raise ValueError(f"MathQA choice parse failed: {row['options']!r}")
        yield {
            "problem": row["Problem"],
            "choices": choices,
            "gold": ["a", "b", "c", "d", "e"].index(row["correct"]),
        }


def _boolq(path: Path) -> Iterable[dict]:
    for row in _read_parquet(path):
        yield {
            "passage": row["passage"],
            "question": row["question"],
            "choices": ["no", "yes"],
            "gold": int(row["label"]),
        }


def _arc_challenge(path: Path) -> Iterable[dict]:
    numeric_labels = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    for row in _read_parquet(path):
        answer = numeric_labels.get(str(row["answerKey"]), str(row["answerKey"]))
        labels = [
            numeric_labels.get(str(label), str(label))
            for label in row["choices"]["label"]
        ]
        yield {
            "question": row["question"],
            "choices": list(row["choices"]["text"]),
            "gold": labels.index(answer),
        }


def _winogrande(path: Path) -> Iterable[dict]:
    for row in _read_parquet(path):
        yield {
            "sentence": row["sentence"],
            "options": [row["option1"], row["option2"]],
            "gold": int(row["answer"]) - 1,
        }


CONVERTERS: dict[str, Callable[[Path], Iterable[dict]]] = {
    "piqa": _piqa,
    "hellaswag": _hellaswag,
    "race": _race,
    "mathqa": _mathqa,
    "boolq": _boolq,
    "arc_challenge": _arc_challenge,
    "winogrande": _winogrande,
}


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return count


def materialize(data_root: Path, raw_paths: dict[str, Path]) -> dict:
    prepared_root = data_root / "prepared"
    manifest = {
        "format_version": 1,
        "task_definition": {
            "project": "EleutherAI/lm-evaluation-harness",
            "commit": TASK_DEFINITION_COMMIT,
            "shots": 0,
            "precision": "bf16",
        },
        "tasks": {},
    }
    for task in PAPER_TASKS:
        output = prepared_root / f"{task}.jsonl"
        count = _write_jsonl_atomic(output, CONVERTERS[task](raw_paths[task]))
        source = MATHQA_SOURCE if task == "mathqa" else HF_SOURCES[task]
        manifest["tasks"][task] = {
            "split": source["split"],
            "examples": count,
            "primary_metric": PRIMARY_METRICS[task],
            "prepared_file": str(output.relative_to(data_root)),
            "prepared_sha256": _sha256(output),
            "source_sha256": source["sha256"],
            **(
                {"source_url": source["url"]}
                if task == "mathqa"
                else {
                    "source_repo": source["repo"],
                    "source_revision": source["revision"],
                    "source_file": source["filename"],
                }
            ),
        }
        print(f"prepared {task}: {count} examples -> {output}")

    manifest_path = data_root / "manifest.json"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=data_root, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(manifest, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(tmp_path, manifest_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return manifest


def verify(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for task, spec in manifest["tasks"].items():
        path = data_root / spec["prepared_file"]
        if not _valid(path, spec["prepared_sha256"]):
            raise RuntimeError(f"prepared checksum mismatch: {task}: {path}")
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        if count != spec["examples"]:
            raise RuntimeError(f"prepared row-count mismatch: {task}: {count}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        manifest = verify(data_root)
    else:
        raw_paths = download_all(data_root / "raw", args.jobs, args.offline)
        manifest = materialize(data_root, raw_paths)
        verify(data_root)
    total = sum(item["examples"] for item in manifest["tasks"].values())
    print(f"paper task data ready: {data_root} ({total} evaluation examples)")


if __name__ == "__main__":
    main()
