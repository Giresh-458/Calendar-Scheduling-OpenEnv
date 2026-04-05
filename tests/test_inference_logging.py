from __future__ import annotations

import re

from inference import build_model_client, main
from inference import log_step
from models import CalendarAction


def test_inference_emits_only_structured_log_lines(monkeypatch, capsys):
    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TASK_IDS", "task_easy,task_medium,task_hard")
    monkeypatch.setenv("MAX_AGENT_STEPS", "8")
    monkeypatch.setenv("BENCHMARK_NAME", "calendar_scheduling_env")

    main()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) >= 9
    assert sum(line.startswith("[START]") for line in lines) == 3
    assert sum(line.startswith("[END]") for line in lines) == 3
    assert all(line.startswith(("[START]", "[STEP]", "[END]")) for line in lines)

    end_lines = [line for line in lines if line.startswith("[END]")]
    assert all(
        re.fullmatch(
            r"\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=.*",
            line,
        )
        for line in end_lines
    )


def test_log_step_preserves_raw_error_text(capsys):
    action = CalendarAction(action_type="cancel_event", event_id=99)
    raw_error = "Cancel rejected because the event_id does not exist."

    log_step(step=1, action=action, reward=-0.5, done=False, error=raw_error)

    line = capsys.readouterr().out.strip()
    assert line.endswith(f"error={raw_error}")


def test_inference_logs_configured_model_name_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TASK_IDS", "task_easy")
    monkeypatch.setenv("MODEL_NAME", "sample-model-name")

    main()

    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line == "[START] task=task_easy env=calendar_scheduling_env model=sample-model-name"


def test_build_model_client_requires_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("MODEL_NAME", "sample-model-name")

    client, model_name = build_model_client()

    assert client is None
    assert model_name == "sample-model-name"


def test_build_model_client_accepts_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("MODEL_NAME", "sample-model-name")

    client, model_name = build_model_client()

    assert client is not None
    assert model_name == "sample-model-name"
