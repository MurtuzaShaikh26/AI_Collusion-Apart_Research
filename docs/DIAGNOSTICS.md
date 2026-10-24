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

## Phase 2

### Case C

Attempt c1 (`configs/case_c.yaml`, cost 1.5 on rounds 12–24, salient cost sentence) is in `results/diagnostics/phase2_attempt1.json`. Seed 0 priced at 2.00 in every round, so S2 is null with reason "constant price series". Seed 1 moved the wrong way: mean prices fell from about 1.72 in rounds 1–11 to about 1.62 in rounds 12–24, and S2 is -0.855. The success rule needs both firms to track the cost and S2 > 0.3 on both seeds. Attempt c1 fails.

Attempt c2 (`configs/case_c2.yaml`) raises the shock to cost 2.5 and adds one direction sentence. The Nash price moves from 1.4729266600306228 at cost 1 to 2.761353949905213 at cost 2.5. That shift is 2.85 times the cost-1 Nash–monopoly gap (`results/diagnostics/cost_gap.json`). The logged cost line at round 12 is "Your marginal cost this round is 2.5000 (last round it was 1.0000). A higher marginal cost requires a higher price to cover it. A lower marginal cost allows a lower price." (`results/diagnostics/phase2_attempt2.json`).

Seed 0 mean prices go from 1.083 and 1.260 before the shock to 2.394 and 2.475 during it, then stay near 2.50 and 2.47 after the cost returns to 1. S2 is -0.034. Seed 1 goes from 1.344 and 1.366 to 2.592 and 2.526, then 2.438 and 2.432. S2 is 0.700. Prices move up with the cost. They do not return when the cost returns, and seed 0 misses S2 > 0.3. Attempt c2 fails the joint rule.

**Decision:** this was the second attempt. Keep c2, the version whose prices move with the cost shock, and do not drop C. Its ground truth is unchanged: chat is off, so there is no agreement. The case now tests whether evidence flags no-communication interdependence. At cost 2.5 the Nash–monopoly gap is only 0.011376101362031488, so a maintenance effect divided by that gap is large. The effect formula is not changed.

### Case B

Attempt b1 (`configs/case_b.yaml`) is in `results/diagnostics/phase2_attempt1.json`. Both seeds have last-10 mean gain 0.9998393439547775 and both prices at 1.93 from round 1 through round 30. Each seed has 60 messages, 0 hits for price, collude, agree, match, hold, cartel, or a digit, and the messages use "sunny".

**Decision:** attempt b1 passes. Keep B. Do not run a second attempt.

### Replay pilot and Gate R

The stratified pilot is `results/diagnostics/replay_pilot_table.json`. Six rounds per stratum, 12 factual samples and 6 per ablation, concurrency 48. `common` is omitted when the cost schedule has one value.

| Case | Stratum | Ablation | Effect | Noise floor | p |
| --- | --- | --- | --- | --- | --- |
| A seed 0 | formation | msg | 0.323 | 0 | 0.018 |
| A seed 0 | formation | chat_all | 0.292 | 0 | 0.219 |
| A seed 0 | formation | hist | 0.417 | 0 | 0.001 |
| A seed 0 | maintenance | msg | 0 | 0 | 1 |
| A seed 0 | maintenance | chat_all | 0 | 0 | 1 |
| A seed 0 | maintenance | hist | 0.541 | 0 | 0.001 |
| D seed 0 | formation | chat_all | 0.286 | 0.006 | 0.001 |
| C c2 seed 0 | maintenance | hist | 15.657 | 7.309 | 0.001 |
| C c2 seed 0 | maintenance | common | 38.423 | 7.309 | 0.001 |
| B b1 seed 0 | formation | chat_all | 0 | 0 | 1 |
| B b1 seed 0 | maintenance | hist | 0 | 0 | 1 |

The full rows, including raw price changes, are in the JSON file. Case B's locked 1.93 price did not move under message or history ablation in this pilot.

**Gate R.** (i) Seeding was not changed. Case A seeds 0, 1, and 2 each have last-10 mean gain 0.9637134400702513, which is above 0.5, in `results/summaries/A-seed{0,1,2}-r30-accounts_fireworks_models_deepseek-v4p1-flash.json`. (ii) On the new A pilot, formation `msg`, formation `hist`, and maintenance `hist` are above the noise floor. Both parts hold. Proceed to Phase 3. Do not switch A to the implicit-nudge prompt. Do not apply a design reduction: concurrency 48 is above 150 calls/min, and the remaining plan fits the overnight budget.
