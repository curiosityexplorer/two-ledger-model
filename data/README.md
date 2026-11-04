# Synthetic datasets

Every file here is produced by `python3 twoledger_sim.py --export-data data` with seed `20261103`. Nothing comes from a production system. Providers (A, B, C), SKUs, list rates, and systems of record (`crm`, `erp1`, `erp2`, `itsm`, `policy`, `claims`, `pay`, `sup`) are generic.

Numbers are written to 6–8 decimal places. Rerunning the script gives the exact values it used in memory, and `verify_data.py` shows the rounding does not change any published result.

## Samples: `evaluation/`, `calibration/`, `training/`

Three independent samples from the same generator, each with two files.

| Folder | Transactions | Usage records | Used for |
|---|---|---|---|
| `evaluation/` | 20,000 | 109,792 | E1, E2, and the E3 test set |
| `calibration/` | 4,000 | 21,804 | Score-to-confidence calibration for probabilistic matching (L4) |
| `training/` | 12,000 | 65,707 | Fitting the E3 forecaster and the per-type cost objectives |

IDs are unique within a sample, not across samples.

### `transactions.csv.gz`: one row per business transaction

| Column | Meaning |
|---|---|
| `bt_id` | Business transaction ID |
| `bt_type` | `O2C` (order-to-cash), `P2P` (procure-to-pay), or `CLAIM` (claim-to-settlement) |
| `tenant` | Tenant, 0–5 |
| `start_hour` | Transaction start, in hours from the start of the 30-day billing period |
| `end_hour` | Time of its last model call |
| `runaway` | 1 if a runaway agent loop was planted in this transaction |
| `n_invocations` | Number of model calls |
| `cost_usd` | Settled list cost of all its calls (ground truth) |

### `usage_records.csv.gz`: one row per model invocation

This is the attribution ledger with its ground truth attached.

| Column | Meaning |
|---|---|
| `record_id` | Invocation ID |
| `true_bt_id` | The transaction this call belongs to. This is the ground truth the correlation spine has to recover. |
| `bt_type`, `tenant` | Type and tenant of that transaction |
| `system` | System of record the calling agent runs in |
| `agent` | Agent role, for example `intake`, `pricing`, `supervisor` |
| `step` | Step index in the transaction template, 0–4 |
| `provider`, `sku` | Model provider and priced offering |
| `t_hour` | Call time in hours from period start. A few calls run slightly past hour 720; E2 counts them in day 30. |
| `input_tokens`, `output_tokens` | Token counts |
| `cost_usd` | Settled list cost of the call |
| `failed` | 1 if the call failed and was retried |
| `coordination` | 1 for supervisor (coordination) calls |

Which records carry trace context, A2A/MCP metadata, or business-event keys is drawn per E1 configuration at run time, so those flags are not stored here. Rerun the script to reproduce them.

## E2 reconciliation: `e2_reconciliation/`

`price_sheet.csv` holds the settled list price per SKU, in USD per million tokens. Each folder `shadow_0pct` … `shadow_8pct` is one run of E2, in which unmetered usage equal to 0, 1, 2, 4, or 8% of metered usage was added to the bill. The attribution ledger for all five runs is `evaluation/usage_records.csv.gz`.

**What the engine sees**

| File | Contents |
|---|---|
| `billing_export.csv` | Simplified FOCUS-style daily bill per SKU. `period` is `current`, or `next` for day 1 of the next period, where usage shifted across the period boundary reappears. Current-period quantities are rounded by the provider to thousands of tokens. `billed_usage_cost_usd` is usage, `commitment_cost_usd` is the idle part of the provisioned-throughput commitment billed evenly per day (B-frontier only), and `billed_cost_usd` is their sum. |
| `rate_card.csv` | The enterprise rate card used to price records on write. `drift_vs_price_sheet` is its relative error against the settled price. |
| `commitment_contract.csv` | The provisioned-pool contract: commitment value, metered consumption from the pool meter, and idle. |

**What the engine never sees**

| File | Contents |
|---|---|
| `planted_ground_truth.csv` | Per SKU: unmetered cost added to the bill, overbilling (3% on C-frontier), under-billing (2% on B-mid) and its amount, and the usage moved into the next period. |

**What the engine produces**

| File | Contents |
|---|---|
| `engine_decomposition.csv` | Per SKU, in USD: billed total, rate drift, price discrepancy (the two-sided check), timing, idle, and the unmetered residual. |
