"""Data and scoring shared by the SEAL math notebooks."""

import json
from pathlib import Path

PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "User: {problem}\nAssistant: <think>"
)


def make_prompts(tokenizer, problems):
    """Tokenize the paper's reasoning prefix without adding a chat template."""
    return [
        {
            "prompt_token_ids": tokenizer.encode(
                PROMPT_TEMPLATE.format(problem=problem), add_special_tokens=True
            )
        }
        for problem in problems
    ]


def load_evaluation_data(path, limit=0):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]
    problems = []
    answers = []
    for record in records:
        is_gsm8k = "question" in record
        problems.append(record["question" if is_gsm8k else "problem"])
        answer = record["answer"]
        # GSM8K stores a worked solution followed by its numeric answer.
        if is_gsm8k:
            answer = answer.rsplit("####", 1)[-1].strip()
        answers.append(answer)
    return problems, answers


def score_outputs(problems, answers, outputs):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    extraction = (ExprExtractionConfig(), LatexExtractionConfig())
    rows = []
    for problem, answer, output in zip(problems, answers, outputs, strict=True):
        completion = output.outputs[0]
        rows.append(
            {
                "problem": problem,
                "answer": answer,
                "output": completion.text,
                "generated_tokens": len(completion.token_ids),
                "correct": bool(
                    verify(
                        parse(f"${answer}$", extraction_config=extraction),
                        parse(completion.text, extraction_config=extraction),
                    )
                ),
            }
        )
    if not rows:
        raise ValueError("The evaluation dataset is empty")
    metrics = {
        "questions": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in rows)
        / len(rows),
    }
    return metrics, rows
