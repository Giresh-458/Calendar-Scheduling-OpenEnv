from __future__ import annotations

import re

from inference import main


def test_inference_emits_only_structured_log_lines(monkeypatch, capsys):
    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
    assert all(re.fullmatch(r"\[END\] success=(true|false) steps=\d+ rewards=.*", line) for line in end_lines)
