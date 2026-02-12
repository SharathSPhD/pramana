"""Unit tests for VyaptiEvaluationRunner answer parsing."""

from pathlib import Path

from pramana.benchmarks.vyapti_runner import VyaptiEvaluationRunner


def _make_runner() -> VyaptiEvaluationRunner:
    project_root = Path(__file__).resolve().parents[3]
    config = {
        "benchmark_path": "data/vyapti_probe/problems.json",
        "solutions_path": "data/vyapti_probe/solutions.json",
    }
    return VyaptiEvaluationRunner(config=config, project_root=project_root)


def test_no_label_does_not_match_cannot_determine() -> None:
    runner = _make_runner()
    response = "Final answer: cannot determine from the given information."
    solution = {"answer": "No, the claim does not hold."}
    assert runner.check_answer(response, {}, solution) is False


def test_no_label_does_not_match_unknown() -> None:
    runner = _make_runner()
    response = "Final answer: unknown at this stage."
    solution = {"answer": "No, the claim does not hold."}
    assert runner.check_answer(response, {}, solution) is False


def test_yes_label_does_not_match_analysis_word() -> None:
    runner = _make_runner()
    response = "My analysis indicates more evidence is needed."
    solution = {"answer": "Yes, the claim holds."}
    assert runner.check_answer(response, {}, solution) is False


def test_empty_solution_answer_returns_false() -> None:
    runner = _make_runner()
    response = "Final answer: yes."
    solution = {"answer": ""}
    assert runner.check_answer(response, {}, solution) is False


def test_yes_answer_is_detected() -> None:
    runner = _make_runner()
    response = "Final answer: Yes, the claim holds."
    solution = {"answer": "Yes, the claim holds."}
    assert runner.check_answer(response, {}, solution) is True


def test_no_answer_is_detected() -> None:
    runner = _make_runner()
    response = "Final answer: No, this is incorrect."
    solution = {"answer": "No, this is incorrect."}
    assert runner.check_answer(response, {}, solution) is True
