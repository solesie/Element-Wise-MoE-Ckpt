"""Pinned zero-shot protocol for the 350M-MoE checkpoint paper."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from lm_eval.base import MultipleChoiceTask, Task, rf
from lm_eval.metrics import mean


TASK_NAMES = (
    "piqa",
    "hellaswag",
    "race",
    "mathqa",
    "boolq",
    "arc_challenge",
    "winogrande",
)
DEFAULT_DATA_ROOT = Path("/home/paul/data/lm_eval_paper_6tasks")


def _data_root() -> Path:
    return Path(os.environ.get("LM_EVAL_PAPER_DATA", DEFAULT_DATA_ROOT)).expanduser()


def _load(task: str) -> list[dict]:
    path = _data_root() / "prepared" / f"{task}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prepared task data: {path}; run "
            "tasks/eval_harness/prepare_paper_tasks.py"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _LocalValidationTask(Task):
    TASK_NAME = ""

    def download(self):
        self.data = _load(self.TASK_NAME)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return iter(self.data)


class _LocalTestTask(_LocalValidationTask):
    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return iter(self.data)


class PiQA(_LocalValidationTask, MultipleChoiceTask):
    VERSION = 1
    TASK_NAME = "piqa"

    def doc_to_text(self, doc):
        return f"Question: {doc['goal']}\nAnswer:"


class HellaSwag(_LocalValidationTask, MultipleChoiceTask):
    VERSION = 1
    TASK_NAME = "hellaswag"

    def doc_to_text(self, doc):
        return doc["query"]


class MathQA(_LocalTestTask, MultipleChoiceTask):
    VERSION = 1
    TASK_NAME = "mathqa"

    def doc_to_text(self, doc):
        return f"Question: {doc['problem']}\nAnswer:"


class BoolQ(_LocalValidationTask, MultipleChoiceTask):
    VERSION = 2
    TASK_NAME = "boolq"

    def doc_to_text(self, doc):
        return f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"

    def process_results(self, doc, results):
        return {"acc": float(np.argmax(results) == doc["gold"])}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class ARCChallenge(_LocalTestTask, MultipleChoiceTask):
    VERSION = 0
    TASK_NAME = "arc_challenge"

    def doc_to_text(self, doc):
        return f"Question: {doc['question']}\nAnswer:"


class WinoGrande(_LocalValidationTask):
    VERSION = 0
    TASK_NAME = "winogrande"

    @staticmethod
    def _partial_context(doc, option):
        blank = doc["sentence"].index("_")
        return doc["sentence"][:blank] + option

    @staticmethod
    def _partial_target(doc):
        blank = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][blank:].strip()

    def doc_to_text(self, doc):
        return self._partial_context(doc, doc["options"][doc["gold"]])

    def doc_to_target(self, doc):
        return self._partial_target(doc)

    def construct_requests(self, doc, ctx):
        target = self._partial_target(doc)
        contexts = ctx.split("\n\n")
        contexts.pop()
        prefix = "\n\n".join(contexts)
        return [
            rf.loglikelihood(
                "\n\n".join(filter(None, (prefix, self._partial_context(doc, option)))),
                target,
            )[0]
            for option in doc["options"]
        ]

    def process_results(self, doc, results):
        return {"acc": float(np.argmax(results) == doc["gold"])}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class RACE(_LocalTestTask):
    VERSION = 3
    TASK_NAME = "race"
    _LETTER_TO_NUM = {"A": 0, "B": 1, "C": 2, "D": 3}

    @classmethod
    def _answer(cls, problem):
        return problem["options"][cls._LETTER_TO_NUM[problem["answer"]]]

    def doc_to_text(self, doc):
        text = "Article: " + doc["article"] + "\n\n"
        for problem in doc["problems"][:-1]:
            if problem["question"][-6:] == "  _  .":
                text += problem["question"][:-5] + self._answer(problem) + "\n"
            else:
                text += "Question: " + problem["question"] + "\n"
                text += "Answer: " + self._answer(problem) + "\n"
        return text + doc["problems"][-1]["question"]

    def doc_to_target(self, doc):
        return " " + self._answer(doc["problems"][-1])

    def construct_requests(self, doc, ctx):
        return [
            rf.loglikelihood(ctx, " " + option)[0]
            for option in doc["problems"][-1]["options"]
        ]

    def process_results(self, doc, results):
        gold = self._LETTER_TO_NUM[doc["problems"][-1]["answer"]]
        return {"acc": float(np.argmax(results) == gold)}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


PAPER_TASK_REGISTRY = {
    "piqa": PiQA,
    "hellaswag": HellaSwag,
    "race": RACE,
    "mathqa": MathQA,
    "boolq": BoolQ,
    "arc_challenge": ARCChallenge,
    "winogrande": WinoGrande,
}


def get_paper_task_dict(names: list[str]) -> dict[str, Task]:
    unknown = sorted(set(names) - set(PAPER_TASK_REGISTRY))
    if unknown:
        raise ValueError(f"unsupported paper task(s): {', '.join(unknown)}")
    return {name: PAPER_TASK_REGISTRY[name]() for name in names}


def protocol_metadata() -> dict:
    path = _data_root() / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
