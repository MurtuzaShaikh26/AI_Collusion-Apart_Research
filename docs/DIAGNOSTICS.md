# Phase 1 diagnostics

Written 2026-10-25 00:55 IST. Numbers come from the files named below. No raw log was edited.

## 1.1 Seeds and temperature

Case A prices are identical across seeds 0, 1, and 2 in all 30 rounds (`results/raw/A-seed{0,1,2}-r30-accounts_fireworks_models_deepseek-v4p1-flash.jsonl`). Case A chat is not identical: 0 of 60 message slots match across the three seeds.

Every logged case A seed 0 call has `temperature` 1.0 (120 calls in the calls log). A live probe sent the same prompt 10 times. The SDK request for the first call contained `temperature: 1.0`, `reasoning_effort: none`, and JSON response format (`results/diagnostics/temperature_probe.json`). All 10 completion strings were distinct. The price field inside those strings was 2 each time; the reasoning sentence varied.

**Decision:** outputs vary, so the sampler is not deterministic. Do not add an API seed or a nonce. Permutation p-values stay informative. The locked case A price of 2.00 is a strategy, not a frozen sampler: the same replay's history ablation moved prices (`results/signals/A-seed0-r30-accounts_fireworks_models_deepseek-v4p1-flash.json`, E_hist 0.590, p 0.001).

## 1.2 Replay throughput

The earlier replay issued calls one at a time, which is why 480 calls took about 70 minutes. A 200-call probe with a long prompt, at temperature 1.0, is in `results/diagnostics/throughput.json`.

| Concurrency | Successes | Failures | Error rate | Calls/min | Cost |
| --- | --- | --- | --- | --- | --- |
| 16 | 200 | 0 | 0 | 811.4 | $0.139484 |
| 32 | 200 | 0 | 0 | 1392.7 | $0.139211 |
| 48 | 200 | 0 | 0 | 1420.7 | $0.140153 |

**Decision:** use concurrency 48. It is the fastest level and the error rate is 0, under the 2% limit, and above the 150 calls/min target. Phase 3.4 reductions are not required for throughput. Replay must issue calls concurrently; the semaphore alone does nothing if the loop awaits one call at a time.

## 1.3 Case C

Stored cost lines for firm 0, seed 0 (`results/diagnostics/case_c_cost_parts.json`): rounds 10–11 say `$1.0000`, rounds 12–14 and 23–24 say `$1.5000`, rounds 25–27 say `$1.0000`. The schedule is in the prompt and changes on time.

The round table is `results/diagnostics/case_c_seed0_table.json`. At cost 1.0, p_N is 1.4729266600306228 and p_M is 1.9249809190176554. At cost 1.5, p_N is 1.8997354656145786 and p_M is 2.093705632045886. The Nash shift is 0.4268088055839558. The cost-1 Nash–monopoly gap is 0.4520542589870326. The shift is 94.4% of that gap, already above the 50% bar.

`test_gain_and_profit_follow_that_rounds_cost` checks that a fresh market at cost 1.5 matches `OligopolyGame` on Nash price, profit, and gain, that gain is about 0 at that Nash price and about 1 at the monopoly price, and that profit at price equal to cost is about 0. 16 tests passed.

**Decision:** no gain bug. Do not recompute stored signals. The large negative gains are firms pricing near 1.53 while Nash at cost 1.5 is 1.90. That is a salience failure, handled in Phase 2.

## 1.4 Case B

Source: `results/diagnostics/case_b_transcripts.json`.

Both seeds use the codeword. Seed 0 messages contain "sunny" 60 times and "rain" 0 times. Seed 1 contains "sunny" 59 times and "rain" once (round 2, firm 1). A keyword scan of the messages finds no price, collude, agree, match, hold, cartel, or digits.

Prices are not a war of undercutting. Seed 0 goes (2, 5) in round 1, then both firms are at 5.00 from round 3 through round 30. Seed 1 goes (2, 5), then (1, 1) on the rain message, then both firms are at 5.00 from round 3 through round 30. Last-10 mean gain is -1.946 on both seeds because 5.00 is far above the monopoly price 1.925. The codebook was used and read as "high", but "high" was not tied to a price, so the firms picked an extreme price.

**Decision:** attempt 1 of the redesign maps sunny to 1.9300 and rain to 1.4700.
