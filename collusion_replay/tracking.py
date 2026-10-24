"""Weights & Biases logging. Disabled unless a run asks for it."""

from __future__ import annotations

import os
import re


class Tracker:
    def __init__(self, enabled: bool, run_id: str, config: dict):
        self.enabled = bool(enabled)
        self.run = None
        self._wandb = None
        if not self.enabled or os.environ.get("WANDB_MODE") == "disabled":
            self.enabled = False
            return
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is missing; refusing to start a run that asked for W&B logging.")
        import wandb

        self._wandb = wandb
        safe_id = re.sub(r"[^A-Za-z0-9]", "", run_id)[:64] or "run"
        self.run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "vibhork11"),
            project=os.environ.get("WANDB_PROJECT", "Apart Research- AI Collusion"),
            name=run_id,
            id=safe_id,
            resume="allow",
            config={k: v for k, v in config.items() if k != "private"},
            tags=[str(config.get("case", "na")), str(config.get("provider", "na"))],
            settings=wandb.Settings(x_service_wait=90),
        )

    def log(self, data: dict, step: int) -> None:
        if self.run is not None:
            self._wandb.log(data, step=step)

    def finish(self, summary: dict | None = None) -> None:
        if self.run is None:
            return
        for key, value in (summary or {}).items():
            if isinstance(value, (int, float, str, bool)):
                self.run.summary[key] = value
        self._wandb.finish()
        self.run = None
