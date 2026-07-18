#!/usr/bin/env bash
# Download DKYoon/SlimPajama-6B and build Megatron mmap datasets.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/.." && pwd)
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

DATA_ROOT=${DATA_ROOT:-/home/paul/data/slimpajama_6b}
HF_REPO=${HF_REPO:-DKYoon/SlimPajama-6B}
RAW_DIR=${RAW_DIR:-"${DATA_ROOT}/raw_hf"}
JSONL_DIR=${JSONL_DIR:-"${DATA_ROOT}/jsonl"}
VOCAB_PATH=${VOCAB_PATH:-"${REPO_ROOT}/examples_deepspeed/rebase/gpt2-vocab.json"}
MERGE_PATH=${MERGE_PATH:-"${REPO_ROOT}/examples_deepspeed/rebase/gpt2-merges.txt"}
WORKERS=${WORKERS:-$(python3 - <<'PY'
import os
print(max(1, min(32, (os.cpu_count() or 4) - 2)))
PY
)}

mkdir -p "${RAW_DIR}" "${JSONL_DIR}" "${DATA_ROOT}"

if [ ! -e "${RAW_DIR}/README.md" ] || ! find "${RAW_DIR}/data" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
    echo "[slimpajama] downloading ${HF_REPO} to ${RAW_DIR}"
    if command -v hf >/dev/null 2>&1; then
        hf download "${HF_REPO}" \
            --repo-type dataset \
            --local-dir "${RAW_DIR}"
    else
        huggingface-cli download "${HF_REPO}" \
            --repo-type dataset \
            --local-dir "${RAW_DIR}"
    fi
else
    echo "[slimpajama] raw HF parquet already present: ${RAW_DIR}"
fi

export RAW_DIR JSONL_DIR
python3 - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

raw = Path(os.environ["RAW_DIR"]) / "data"
jsonl = Path(os.environ["JSONL_DIR"])
jsonl.mkdir(parents=True, exist_ok=True)

splits = {
    "train": sorted(raw.glob("train-*.parquet")),
    "valid": sorted(raw.glob("validation-*.parquet")),
}

for split, files in splits.items():
    if not files:
        raise SystemExit(f"missing parquet files for split={split} under {raw}")
    out = jsonl / f"slimpajama_6b_{split}.jsonl"
    done = out.with_suffix(out.suffix + ".done")
    if done.exists() and out.exists() and out.stat().st_size > 0:
        print(f"[slimpajama] jsonl already present: {out}")
        continue
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        n = 0
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    if text:
                        fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        n += 1
            print(f"[slimpajama] converted {path.name}; rows={n}", flush=True)
    tmp.replace(out)
    done.write_text(str(n) + "\n")
    print(f"[slimpajama] wrote {out}; rows={n}", flush=True)
PY

build_split() {
    local split="$1"
    local input="${JSONL_DIR}/slimpajama_6b_${split}.jsonl"
    local prefix="${DATA_ROOT}/slimpajama_6b_${split}"
    if [ -e "${prefix}_text_document.idx" ] && [ -e "${prefix}_text_document.bin" ]; then
        echo "[slimpajama] mmap already present: ${prefix}_text_document"
        return
    fi
    echo "[slimpajama] preprocessing ${split} with ${WORKERS} workers"
    python3 "${REPO_ROOT}/tools/preprocess_data.py" \
        --input "${input}" \
        --json-keys text \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file "${VOCAB_PATH}" \
        --merge-file "${MERGE_PATH}" \
        --append-eod \
        --workers "${WORKERS}" \
        --output-prefix "${prefix}" \
        --dataset-impl mmap
}

build_split train
build_split valid

echo "[slimpajama] ready:"
echo "  DATA_PATH=${DATA_ROOT}/slimpajama_6b_train_text_document"
echo "  VALID_DATA_PATH=${DATA_ROOT}/slimpajama_6b_valid_text_document"
