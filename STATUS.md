# STATUS  (last updated: 2026-10-25 01:12 IST)

## Overnight progress

- Orchestrator PID: 67412. Process name python. Started 2026-10-25 01:11:46 IST. Log: `logs/overnight.log`.
- Phase: 3, model 1 trajectories, stratified replays, and judges. Pre-registration commit `64b772a` at 2026-10-25 01:11:33 +0530.
- Cases kept: A, B (`configs/case_b.yaml`), C (`configs/case_c2.yaml`), D. No design reduction.
- Spend at launch: $2.270665 (`results/spend.json`). Tonight cap $25.746654. Hard cap $40.
- Estimated model-1 calls still to launch at start: 17720.

<!-- HISTORY -->

# STATUS  (last updated: 2026-10-24 22:17 IST)

Stage: 4 replay pilot done   Overall: on track
Done this stage: Replay finished for case A seed 0 and case C seed 0. Each used 8 rounds, 12 factual samples and 6 samples per ablation, 480 new calls. Effects are fractions of that round's monopoly-minus-Nash price gap. Case A: E_msg 0 (p=1), E_hist 0.590 (p=0.001), E_common 0 (p=1), noise floor 0. Case C: E_msg 0.056 (p=0.50), E_hist 0.886 (p=0.001), E_common 0.060 (p=0.025), noise floor 0.046. Case C has chat off, so the message ablation is not a communication test. Partner-price history moves both replays. Message text does not move the locked case A price of 2.00. Files: `results/signals/` and `results/RESULTS.md`.
Key results so far: Case A gain 0.964 on 3/3 seeds. B, C, and D pilots do not clear 0.5. Replay says the case A price survives replacing the chat and the cost line, and moves when partner prices are replaced.
Running in background: nothing.
Spend so far: $0.746654 (1871 calls, 1890447 prompt tokens, 150188 completion tokens; `results/spend.json`, 2026-10-24 22:16 IST). The two replays added about $0.40.
NEXT: Pre-register before any full multi-seed replay. Do not start the large replay until that file is written.
NEED FROM MURT:
1. Nothing blocking. The laptop can sleep.
2. Rotate the Fireworks key that was pasted in chat, and keep the new value only in `.env`.
