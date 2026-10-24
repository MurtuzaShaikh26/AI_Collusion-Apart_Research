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

## 2026-10-25T02:28:16+05:30 — Phase 6

1. What changed. Phase 1 found varying samples at temperature 1.0, so no API seed was added. Case C's gain formula already uses the round cost. Case B's first codebook priced at 5.00. Phase 2 decisions and any prompt changes are in `docs/DIAGNOSTICS.md`. Deviations from the pre-registration, including design reductions, are listed here and are not written into `docs/PREREG.md`.
   Reductions used: none.
   Stop note: completed the planned phases.
2. Final cases: A (configs/case_a.yaml), B (configs/case_b.yaml), C (configs/case_c2.yaml), D (configs/case_d.yaml). Seeds [0, 1, 2, 3, 4, 5, 6]. Train [0, 1, 2, 3]. Test [4, 5, 6]. Conditions ['msg', 'chat_all', 'hist', 'common']. Rounds per stratum 6. Factual samples 12.
3. Hypotheses:
   - H1: not supported. mean S3 D=7.0000 C=0.0000; D flagged 0/3 at threshold 0.6072; B missed 3/3 (`results/analysis/hypotheses.json`).
   - H2: supported. S2 AUROC 0.6806 CI [0.3438, 0.9444] (`results/analysis/hypotheses.json`).
   - H3: not supported. A 2/7; mean E_chat_all A=1.0461 D=0.4535; B 0/7; B=0.0018873278731770012 (`results/analysis/hypotheses.json`).
   - H4: supported. E_hist > E_msg in 7/7 A trajectories (`results/analysis/hypotheses.json`).
   - H5: not supported. replay AUROC 0.75 S3 AUROC 0.5972222222222222; FPR {'S3': 0.0, 'S2': 0.5, 'replay': 0.5, 'all': 0.0} (`results/analysis/hypotheses.json`).
4. Spend tonight is the cumulative ledger minus the $0.746654 already spent before the overnight probes. Cumulative spend is $9.335759 (35574 calls, `results/spend.json`). Tonight cap $25.746654. Hard cap $40.
5. Model 2: model 2 completed with frozen phase-4 detectors. What is left is whatever that line says is unfinished, plus the written report. Murt needs to decide only if a hypothesis is inconclusive because a run stopped, or if the Fireworks key still needs rotating. No question is blocking the files already written.

Orchestrator PID: 17484.

Analysis files: `results/analysis/heatmap.svg`, `results/analysis/detectors.json`, `results/analysis/loco.json`, `results/analysis/formation_example.svg`, `results/RESULTS.md`.
