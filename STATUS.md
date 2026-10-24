# STATUS  (last updated: 2026-10-24 19:23 IST)

Stage: 2 / harness (mock)   Overall: on track
Done this stage: Plan is in `docs/PLAN.md`. Harness, mock provider, and pytest are in place. 12 tests passed on the mock provider (prompt round-trip, ablations, Nash/monopoly vs vendored OligopolyGame, deterministic scripted proposer, parse retries, replay effect formula, end-to-end game + replay + signals). No paid calls. `uv sync` finished (Python 3.12, wandb 0.31.0). A mock smoke run is on the dashboard: https://forge.coreweave.com/wandb/vibhork11/Apart%20Research-%20AI%20Collusion/runs/uz0eqery — logged in as `murtuzashaikh-2023` / entity `vibhork11`. That run is a wiring check, not an experiment. The system Python 3.14 W&B client (0.25.1) hung; use `.venv`.
Key results so far: no experimental numbers yet. Test suite: 12 passed (`python -m pytest`). Spend ledger not opened.
Running in background: `uv sync` (project venv, Python 3.12). Still downloading wheels. Tests already passed on the system Python 3.14 install.
Spend so far: $0 (no Fireworks calls)
NEXT: Fireworks model list, 10-call benchmark, model table, then wait for approval before the positive control.
NEED FROM MURT:
1. Create a Fireworks key at https://app.fireworks.ai → API Keys and paste it into `.env` as `FIREWORKS_API_KEY=` yourself. Do not paste the key into chat.
2. Nothing else is blocking. Modal is already signed in as profile `ronitk964` (token file present; the `modal` command is not on PATH yet, so the CLI install is next). Ollama is installed, but `qwen2.5:7b-instruct` is not pulled. Optional, only if you want the local benchmark: `ollama pull qwen2.5:7b-instruct`.
