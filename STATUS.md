# STATUS  (last updated: 2026-10-25T02:28:16+05:30)

## Morning summary

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
