"""Unit tests for shortcut detection helper utilities."""


def test_build_baseline_prompt_includes_problem() -> None:
    from pramana.application.evaluation.ablation import build_baseline_prompt

    prompt = build_baseline_prompt("Test problem")

    assert "Test problem" in prompt
    assert "Samshaya" not in prompt


def test_build_nyaya_prompt_includes_template() -> None:
    from pramana.application.evaluation.ablation import build_nyaya_prompt

    prompt = build_nyaya_prompt(
        problem="Test problem",
        format_instructions="INSTRUCTIONS",
        format_template="## Samshaya (Doubt Analysis)",
    )

    assert "INSTRUCTIONS" in prompt
    assert "## Samshaya (Doubt Analysis)" in prompt


def test_extract_answer_prefers_final_answer() -> None:
    from pramana.application.evaluation.ablation import extract_answer_from_output

    output = "Some text\nFinal Answer: Bob has the dog\nMore text"
    assert extract_answer_from_output(output) == "Bob has the dog"


def test_extract_answer_falls_back_to_last_line() -> None:
    from pramana.application.evaluation.ablation import extract_answer_from_output

    output = "Reasoning line\nConclusion line"
    assert extract_answer_from_output(output) == "Conclusion line"
