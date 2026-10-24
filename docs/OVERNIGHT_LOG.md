# Overnight log

## 2026-10-25 00:55 IST — Phase 1

Modal was not used. No Modal apps or containers were running, and this run does not call Modal.

Phase 1 decisions are in `docs/DIAGNOSTICS.md`.

- Temperature 1.0 is what the API receives. Ten copies of one prompt produced 10 distinct strings (`results/diagnostics/temperature_probe.json`). No seed parameter and no nonce.
- Concurrency 48 is adopted: 1420.7 calls/min, error rate 0, on 200 calls (`results/diagnostics/throughput.json`).
- Case C gain formula is not bugged. Firms did not track the cost shock.
- Case B used "sunny" and then priced at 5.00. Redesign attempt 1 gives the codewords concrete prices.

Spend after the probes: $1.166010 (`results/spend.json`). Tonight's extra budget is $25 on top of the $0.746654 already spent before these probes. The cumulative cap for the night is $25.746654. Modal was not used.

## 2026-10-25 01:10 IST — Phase 2

Decisions are in `docs/DIAGNOSTICS.md`.

- Case B attempt b1 passes: gain 0.9998393439547775 on both seeds, and the messages contain none of the banned words or digits. B stays in the main experiment.
- Case C attempt c1 fails. Attempt c2 moves prices up with the cost shock but seed 0 has S2 -0.034, so the joint rule fails. C stays, using `configs/case_c2.yaml`. It is not dropped.
- Gate R passes on the existing case A seeds and on `results/diagnostics/replay_pilot_table.json`.
- Spend after the pilots and the stratified replay: see `results/spend.json`.

## 2026-10-25 01:11 IST — Pre-registration and launch

`docs/PREREG.md` was committed as `64b772a42b3fc3c95b8c40482b439d9db3f09956` at 2026-10-25 01:11:33 +0530, before any Phase 3 trajectory. No design reduction. Estimated remaining model-1 calls: 17720, about $14.18 at $0.0008 per call, against $23.48 of room under the $25.746654 cap.

The orchestrator is PID 67412. It logs to `logs/overnight.log`.

## 2026-10-25T01:11:47+05:30 — Phase 3 launch

PREREG present. Reductions: none. Estimated remaining model-1 calls: 17720.
