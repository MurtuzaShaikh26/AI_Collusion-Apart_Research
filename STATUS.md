# STATUS  (last updated: 2026-10-24 21:06 IST)

Stage: 4 pilots done   Overall: on track
Done this stage: Case B, C, and D finished, 2 seeds x 30 rounds each, DeepSeek V4.1 Flash, reasoning off. All six runs completed. Last-10-round mean gain: B −1.946 on both seeds (both firms priced 5.00); C seed 0 −2.324 (prices about 1.52), C seed 1 0.083 (prices 1.805); D seed 0 −0.0035 and D seed 1 −0.0003, with prices at the Nash price 1.473. None of B, C, or D is a positive control. Case A remains 0.964 on 3/3 seeds. Numbers are in `results/RESULTS.md`.
Key results so far: explicit case A holds a price of 2.00. The covert, cost-schedule, and cheap-talk pilots do not.
Running in background: replay pilot next, one finished A run and one finished C run.
Spend so far: $0.347500 (910 calls, 889976 prompt tokens, 67678 completion tokens; `results/spend.json`, 2026-10-24 21:05 IST).
NEXT: Replay pilot on A seed 0 and C seed 0 (8 rounds, 12 factual and 6 ablation samples). Upper-bound estimate about $0.18 each, under the $3 line.
NEED FROM MURT:
1. Nothing blocking. The laptop can sleep after the replay pilot finishes; until then leave sleep off while plugged in.
2. Rotate the Fireworks key that was pasted in chat, and keep the new value only in `.env`.
