#!/usr/bin/env python3
"""Check the published synthetic data against the published results.

Rebuilds the headline numbers from the CSV files alone, without running the
simulation: sample sizes, per-transaction costs, and the E2 reconciliation
(the unmetered residual recovered from the billing export, the price sheet and
the metered usage records). Exits with an error if anything disagrees with
results/results.json.

    python3 verify_data.py                       # uses data/ and results/results.json
    python3 verify_data.py --data DIR --results FILE
"""
import argparse, csv, gzip, json, sys
import numpy as np


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results/results.json")
    a = ap.parse_args(argv)
    D, res = a.data, json.load(open(a.results))

    tx = read(f"{D}/evaluation/transactions.csv.gz")
    ur = read(f"{D}/evaluation/usage_records.csv.gz")
    print(f"evaluation sample: {len(tx)} transactions, {len(ur)} usage records")
    assert len(tx) == res["n_transactions"] and len(ur) == res["n_records"], "sample sizes differ from results.json"

    by_bt = {}
    for r in ur:
        by_bt[r["true_bt_id"]] = by_bt.get(r["true_bt_id"], 0.0) + float(r["cost_usd"])
    gap = max(abs(by_bt[t["bt_id"]] - float(t["cost_usd"])) for t in tx)
    assert gap < 1e-6, "transaction cost does not equal the sum of its usage records"
    print(f"transaction costs equal the sum of their usage records (max gap {gap:.1e} USD)")

    sheet = {r["sku"]: (float(r["input_usd_per_mtok"]), float(r["output_usd_per_mtok"]))
             for r in read(f"{D}/e2_reconciliation/price_sheet.csv")}
    days = 30
    led = {sk: (np.zeros(days), np.zeros(days)) for sk in sheet}
    for r in ur:  # the attribution ledger: metered calls, bucketed by billing day
        d = min(int(float(r["t_hour"]) // 24), days - 1)
        led[r["sku"]][0][d] += int(r["input_tokens"])
        led[r["sku"]][1][d] += int(r["output_tokens"])

    for lvl, e2 in zip(["0", "1", "2", "4", "8"], res["E2"]):
        folder = f"{D}/e2_reconciliation/shadow_{lvl}pct"
        bill = read(f"{folder}/billing_export.csv")
        truth = {r["sku"]: r for r in read(f"{folder}/planted_ground_truth.csv")}
        tot_billed = recovered = planted = 0.0
        for sk, (ri, ro) in sheet.items():
            cur = sorted((r for r in bill if r["sku"] == sk and r["period"] == "current"), key=lambda r: int(r["day"]))
            nxt = next(r for r in bill if r["sku"] == sk and r["period"] == "next")
            b_in = np.array([float(r["billed_input_tokens"]) for r in cur])
            b_out = np.array([float(r["billed_output_tokens"]) for r in cur])
            tot_billed += sum(float(r["billed_cost_usd"]) for r in cur)
            l_in, l_out = led[sk]
            # usage missing on the last day that reappears on day 1 of the next period
            t_in = min(max(l_in[-1] - b_in[-1], 0.0), float(nxt["billed_input_tokens"]))
            t_out = min(max(l_out[-1] - b_out[-1], 0.0), float(nxt["billed_output_tokens"]))
            residual = ((b_in * ri + b_out * ro).sum() - (l_in * ri + l_out * ro).sum() + t_in * ri + t_out * ro) / 1e6
            if float(truth[sk]["underbilling_pct"]) == 0.0:  # the under-billed SKU is reported separately, as in the paper
                recovered += residual
                planted += float(truth[sk]["planted_unmetered_cost_usd"])
        rec_pct, pl_pct = 100 * recovered / tot_billed, 100 * planted / tot_billed
        print(f"E2 shadow {lvl}%: planted {pl_pct:.2f}% of bill, recovered {rec_pct:.2f}% "
              f"(results.json: {e2['planted_pct_of_bill']:.2f}% / {e2['recovered_pct_of_bill']:.2f}%)")
        assert abs(rec_pct - e2["recovered_pct_of_bill"]) < 1e-3 and abs(pl_pct - e2["planted_pct_of_bill"]) < 1e-3, \
            f"E2 shadow {lvl}% does not match results.json"
    print("OK: the published data reproduces the published results")


if __name__ == "__main__":
    sys.exit(main())
