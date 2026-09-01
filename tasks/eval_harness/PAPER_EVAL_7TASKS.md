# Seven-task paper evaluation

This path evaluates the final BF16 MoE weights on PIQA, HellaSwag, RACE-high,
MathQA, BoolQ, ARC-Challenge, and WinoGrande. The primary table is zero-shot
and reports a percentage for each task.

## One-time preparation

The training environment is not modified.  lm-eval 0.2.0 is installed into a
small overlay under `$HOME/.cache`, while task data is converted to explicit
JSONL files under `$HOME/data`.

```bash
tasks/eval_harness/setup_paper_eval_env.sh
.venv/bin/python tasks/eval_harness/prepare_paper_tasks.py --jobs 6
```

The download script pins every Hugging Face repository revision and writes
source/output checksums and row counts to
`$HOME/data/lm_eval_paper_6tasks/manifest.json`.  Evaluation does not use
the Hugging Face datasets library and can run offline.

## Evaluation

Use the same TP1/PP2/EP4 topology as the final training run.  The checkpoint
argument may be the model-only DeepSpeed root or its `global_step10000`
directory.  Inference remains BF16; the old example's `--eval_fp32` flag is
not used.

```bash
tasks/eval_harness/run_paper_eval.sh \
  /persistent/checkpoints/proposal/bf16_model \
  results/proposal_downstream_7tasks.json
```

The checkpoint may be either the `bf16_model` root containing `latest` or the
concrete `global_step10000` directory.  The wrapper always uses the tokenizer
and DeepSpeed config bundled by the terminal model-only save.

The wrapper writes the full lm-eval JSON plus a compact `*.percent.json` file.
Primary metrics are PIQA/HellaSwag/MathQA/ARC-Challenge `acc_norm` and
RACE/BoolQ/WinoGrande `acc`. RACE uses the official `high/test` grouped
passage definition.

For a short plumbing smoke test only, use for example
`EVAL_LIMIT=10 TASKS=piqa`. Published results must leave `EVAL_LIMIT` unset and
use all examples recorded in the manifest.
