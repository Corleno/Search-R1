import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.reprompt_templates import resolve_self_distillation_templates


def _base_cfg(**overrides):
    cfg = {
        "reprompt_template": "{prompt}{solution}{feedback}\n\nSolve again.",
        "solution_template": "\nSolution:\n{successful_previous_attempt}",
        "feedback_template": "\nFeedback:\n{feedback_raw}",
        "reprompt_template_file": None,
        "solution_template_file": None,
        "feedback_template_file": None,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_inline_templates_used_when_no_files(tmp_path):
    cfg = _base_cfg()
    templates = resolve_self_distillation_templates(cfg)

    assert templates["reprompt_template"] == cfg.reprompt_template
    assert templates["solution_template"] == cfg.solution_template
    assert templates["feedback_template"] == cfg.feedback_template


def test_file_contents_override_inline_templates(tmp_path):
    reprompt_file = tmp_path / "reprompt.txt"
    solution_file = tmp_path / "solution.txt"
    feedback_file = tmp_path / "feedback.txt"
    reprompt_file.write_text("Q: {prompt}\n{solution}{feedback}", encoding="utf-8")
    solution_file.write_text("Peer: {successful_previous_attempt}", encoding="utf-8")
    feedback_file.write_text("Note: {feedback_raw}", encoding="utf-8")

    cfg = _base_cfg(
        reprompt_template_file=str(reprompt_file),
        solution_template_file=str(solution_file),
        feedback_template_file=str(feedback_file),
    )
    templates = resolve_self_distillation_templates(cfg)

    assert templates["reprompt_template"] == "Q: {prompt}\n{solution}{feedback}"
    assert templates["solution_template"] == "Peer: {successful_previous_attempt}"
    assert templates["feedback_template"] == "Note: {feedback_raw}"


def test_missing_template_file_raises(tmp_path):
    missing = tmp_path / "missing.txt"
    cfg = _base_cfg(reprompt_template_file=str(missing))

    with pytest.raises(FileNotFoundError, match="reprompt_template_file not found"):
        resolve_self_distillation_templates(cfg)


def test_file_loaded_templates_support_format_placeholders(tmp_path):
    reprompt_file = tmp_path / "reprompt.txt"
    solution_file = tmp_path / "solution.txt"
    feedback_file = tmp_path / "feedback.txt"
    reprompt_file.write_text("{prompt}{solution}{feedback}", encoding="utf-8")
    solution_file.write_text("\nSol: {successful_previous_attempt}", encoding="utf-8")
    feedback_file.write_text("\nFb: {feedback_raw}", encoding="utf-8")

    templates = resolve_self_distillation_templates(_base_cfg(
        reprompt_template_file=str(reprompt_file),
        solution_template_file=str(solution_file),
        feedback_template_file=str(feedback_file),
    ))

    solution_section = templates["solution_template"].format(
        successful_previous_attempt="answer text"
    )
    feedback_section = templates["feedback_template"].format(feedback_raw="try again")
    reprompt_text = templates["reprompt_template"].format(
        prompt="What is 2+2?",
        solution=solution_section,
        feedback=feedback_section,
    )

    assert "What is 2+2?" in reprompt_text
    assert "answer text" in reprompt_text
    assert "try again" in reprompt_text
