# Overnight log

## 2026-10-25 00:55 IST — Phase 1

Modal was not used. No Modal apps or containers were running, and this run does not call Modal.

Phase 1 decisions are in `docs/DIAGNOSTICS.md`.

- Temperature 1.0 is what the API receives. Ten copies of one prompt produced 10 distinct strings (`results/diagnostics/temperature_probe.json`). No seed parameter and no nonce.
- Concurrency 48 is adopted: 1420.7 calls/min, error rate 0, on 200 calls (`results/diagnostics/throughput.json`).
- Case C gain formula is not bugged. Firms did not track the cost shock.
- Case B used "sunny" and then priced at 5.00. Redesign attempt 1 gives the codewords concrete prices.

Spend after the probes: $1.166010 (`results/spend.json`). Tonight's extra budget is $25 on top of the $0.746654 already spent before these probes.
