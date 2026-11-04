#!/usr/bin/env python3
"""
Two-Ledger Model — simulation study on synthetic data with known ground truth.

All data here is generated. Every usage record carries its TRUE business
transaction, so attribution accuracy, reconciliation recovery, and gating
effects are measured against known truth. Nothing is estimated from, or
represents, any production system. Fixed seed: results are reproducible.

Experiments
  E1  Correlation spine: coverage and accuracy as trace propagation degrades,
      versus a trace-only baseline (observability alone).
  E2  Reconciliation: does the decomposed residual recover planted unmetered
      ("shadow AI") spend, and does the two-sided check flag provider billing
      errors?
  E3  Mid-flight gating: tail-cost reduction and false-positive rate of
      forecast-based preemptive gating under runaway agent loops.

Usage
  python3 twoledger_sim.py                              # outputs next to this script
  python3 twoledger_sim.py --out results --export-data data

Outputs: results.json, table_e1.tex, table_e2.tex, table_e3.tex, and
fig_e1/e2/e3 as PDF and PNG. With --export-data DIR, every synthetic dataset
the run uses is also written to DIR as CSV (see data/README.md in the
repository). The export draws no random numbers, so results are identical
with or without it.
"""
import argparse, csv, gzip, io, json, math, os, sys
import numpy as np

SEED = 20261103
rng = np.random.default_rng(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Synthetic estate
# ----------------------------------------------------------------------------
# Providers / SKUs: settled list rates per million tokens (input, output).
SKUS = {
    "A-frontier": (2.00, 8.00),
    "B-frontier": (3.00, 15.00),
    "C-frontier": (1.25, 10.00),
    "B-mid":      (0.80, 3.20),
}
PROVIDER_OF = {"A-frontier": "A", "B-frontier": "B", "C-frontier": "C", "B-mid": "B"}

# Transaction templates: (system, agent, sku, mean_in_tokens, mean_out_tokens, p_fail)
TEMPLATES = {
    "O2C": [
        ("crm",    "intake",    "A-frontier",  6000, 1100, 0.03),
        ("erp1",   "pricing",   "B-frontier",  9500, 1900, 0.05),
        ("erp2",   "order",     "C-frontier",  7500, 1450, 0.12),
        ("itsm",   "exception", "A-frontier", 12000, 2600, 0.08),
        ("sup",    "supervisor","B-mid",       5000,  800, 0.01),
    ],
    "P2P": [
        ("erp1",   "requisition","A-frontier", 5000,  900, 0.03),
        ("erp1",   "approval",   "B-mid",      3000,  500, 0.02),
        ("erp2",   "receiving",  "C-frontier", 6000, 1200, 0.06),
        ("erp2",   "invoice-match","B-frontier",8000, 1500, 0.10),
        ("sup",    "supervisor", "B-mid",      4000,  700, 0.01),
    ],
    "CLAIM": [
        ("crm",    "claim-intake","A-frontier", 9000, 1600, 0.04),
        ("policy", "coverage",    "C-frontier", 7000, 1300, 0.05),
        ("policy", "fraud-screen","B-frontier", 11000,2200, 0.07),
        ("claims", "adjudication","A-frontier", 10000,2400, 0.09),
        ("pay",    "payment",     "B-mid",      3500,  600, 0.02),
    ],
}
# Which systems expose business keys through event feeds (L3 candidates).
KEYED_SYSTEMS = {"crm", "erp1", "erp2", "itsm", "policy", "claims", "pay"}
TENANTS = 6
PERIOD_HOURS = 24 * 30  # one billing month


def settled_cost(sku, tin, tout):
    ri, ro = SKUS[sku]
    return (tin * ri + tout * ro) / 1e6


def gen_transactions(n, p_runaway=0.05, loop_mean=9):
    """Generate transactions and their invocation records with ground truth."""
    types = list(TEMPLATES)
    recs, txs = [], []
    rid = 0
    for b in range(n):
        ttype = types[rng.integers(len(types))]
        tenant = int(rng.integers(TENANTS))
        start = rng.uniform(0, PERIOD_HOURS)
        t = start
        runaway = rng.random() < p_runaway
        loop_step = int(rng.integers(1, 4)) if runaway else -1
        steps = TEMPLATES[ttype]
        tx = {"bt": b, "type": ttype, "tenant": tenant, "start": start, "runaway": runaway, "recs": []}
        for si, (system, agent, sku, mi, mo, pf) in enumerate(steps):
            # optional step: ITSM exception only sometimes
            if agent == "exception" and rng.random() > 0.35:
                continue
            reps = 1
            if runaway and si == loop_step:
                reps += int(rng.geometric(1.0 / loop_mean))
            for r in range(reps):
                # retries
                attempts = 1 + int(rng.random() < pf)
                for a in range(attempts):
                    tin = int(rng.lognormal(math.log(mi), 0.25))
                    tout = int(rng.lognormal(math.log(mo), 0.30))
                    t += rng.uniform(0.05, 0.6)  # hours between calls
                    failed = (attempts == 2 and a == 0)
                    recs.append({
                        "id": rid, "bt": b, "type": ttype, "tenant": tenant, "system": system,
                        "agent": agent, "sku": sku, "t": t, "tin": tin, "tout": tout,
                        "cost": settled_cost(sku, tin, tout), "failed": failed, "step": si,
                        "coord": agent == "supervisor",
                    })
                    tx["recs"].append(rid)
                    rid += 1
        tx["end"] = t
        txs.append(tx)
    return txs, recs


# ----------------------------------------------------------------------------
# Correlation spine
# ----------------------------------------------------------------------------
def l4_score(rec, tx, tau=1.0):
    """Weighted similarity: temporal proximity (decay), tenant match, agent/stage plausibility,
    and position plausibility (where in the transaction's lifetime this call falls)."""
    span = max(tx["end"] - tx["start"], 0.1)
    # expected position of this agent's step within the template
    steps = TEMPLATES[tx["type"]]
    pos = [k for k, st in enumerate(steps) if st[1] == rec["agent"]]
    agent_term = 1.0 if pos else 0.0
    if pos:
        exp_t = tx["start"] + span * (pos[0] + 0.5) / len(steps)
        dt = abs(rec["t"] - exp_t)
    else:
        dt = abs(rec["t"] - 0.5 * (tx["start"] + tx["end"]))
    time_term = math.exp(-dt / tau)
    tenant_term = 1.0 if rec["tenant"] == tx["tenant"] else 0.0
    inside = 1.0 if tx["start"] <= rec["t"] <= tx["end"] else 0.0
    return 0.45 * time_term + 0.25 * tenant_term + 0.15 * agent_term + 0.15 * inside


def l4_candidates(rec, txs_sorted, starts, slack=1.5):
    """Transactions active around the record time (same tenant preferred)."""
    lo = np.searchsorted(starts, rec["t"] - 12.0)
    hi = np.searchsorted(starts, rec["t"] + 0.5)
    cands = [tx for tx in txs_sorted[lo:hi] if tx["start"] - slack <= rec["t"] <= tx["end"] + slack]
    return cands


def calibrate_l4(txs, recs, n_cal=6000, nbins=12):
    """Empirical score -> confidence mapping from a calibration sample with known truth
    (quantile bins of the chosen-candidate score; monotone non-decreasing precision)."""
    txs_sorted = sorted(txs, key=lambda x: x["start"])
    starts = np.array([tx["start"] for tx in txs_sorted])
    idx = rng.choice(len(recs), size=min(n_cal, len(recs)), replace=False)
    scores, correct = [], []
    for i in idx:
        rec = recs[i]
        cands = l4_candidates(rec, txs_sorted, starts)
        if not cands:
            continue
        best = max(cands, key=lambda tx: l4_score(rec, tx))
        scores.append(l4_score(rec, best)); correct.append(best["bt"] == rec["bt"])
    scores, correct = np.array(scores), np.array(correct, dtype=float)
    edges = np.quantile(scores, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    prec = np.array([correct[(scores >= edges[k]) & (scores < edges[k + 1])].mean() for k in range(nbins)])
    prec = np.maximum.accumulate(np.nan_to_num(prec, nan=0.0))
    return edges, prec


def conf_from_score(s, edges, prec):
    k = int(np.searchsorted(edges, s, side="right")) - 1
    k = min(max(k, 0), len(prec) - 1)
    return float(prec[k])


def run_spine(txs, recs, p_trace, p_proto, p_event, p_collision, floor, calib):
    """Assign every record; return per-record (level, assigned_bt, conf)."""
    txs_sorted = sorted(txs, key=lambda x: x["start"])
    starts = np.array([tx["start"] for tx in txs_sorted])
    edges, prec = calib
    out = []
    # instrumentation draws
    has_trace = rng.random(len(recs)) < p_trace
    has_proto = rng.random(len(recs)) < p_proto
    event_ok = rng.random(len(recs)) < p_event
    collide = rng.random(len(recs)) < p_collision
    resolve = rng.random(len(recs)) < 0.7  # scope narrowing resolves most collisions
    for i, rec in enumerate(recs):
        if has_trace[i]:
            out.append(("L1", rec["bt"], 1.0)); continue
        if has_proto[i]:
            out.append(("L2", rec["bt"], 1.0)); continue
        if rec["system"] in KEYED_SYSTEMS and event_ok[i]:
            if not collide[i] or resolve[i]:
                out.append(("L3", rec["bt"], 0.98)); continue
            # unresolved collision -> demote to L4
        cands = l4_candidates(rec, txs_sorted, starts)
        if not cands:
            out.append(("POOL", None, 0.0)); continue
        best = max(cands, key=lambda tx: l4_score(rec, tx))
        c = conf_from_score(l4_score(rec, best), edges, prec)
        if c >= floor:
            out.append(("L4", best["bt"], c))
        else:
            out.append(("POOL", None, c))
    return out


def summarize(recs, assign):
    cost = np.array([r["cost"] for r in recs])
    tot = cost.sum()
    lv = np.array([a[0] for a in assign])
    assigned = lv != "POOL"
    correct = np.array([a[1] == r["bt"] for a, r in zip(assign, recs)])
    res = {
        "coverage": float(assigned.mean()),
        "cost_coverage": float(cost[assigned].sum() / tot),
        "accuracy": float(correct[assigned].mean()) if assigned.any() else float("nan"),
        "cost_accuracy": float(cost[assigned & correct].sum() / cost[assigned].sum()) if assigned.any() else float("nan"),
        "pooled_share": float(cost[~assigned].sum() / tot),
        "levels": {L: float((lv == L).mean()) for L in ["L1", "L2", "L3", "L4", "POOL"]},
        "mis_cost_share": float(cost[assigned & ~correct].sum() / tot),
    }
    return res


# ----------------------------------------------------------------------------
# E1
# ----------------------------------------------------------------------------
def experiment_1(txs, recs, calib, floor=0.8):
    rows = []
    for p_trace in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        a = run_spine(txs, recs, p_trace, p_proto=0.5, p_event=0.85, p_collision=0.05, floor=floor, calib=calib)
        s_ = summarize(recs, a)
        rows.append({"sweep": "p_trace", "p_trace": p_trace, "p_event": 0.85, **s_, "baseline_coverage": s_["levels"]["L1"]})
    for p_event in [1.0, 0.5, 0.0]:
        a = run_spine(txs, recs, 0.4, p_proto=0.5, p_event=p_event, p_collision=0.05, floor=floor, calib=calib)
        s_ = summarize(recs, a)
        rows.append({"sweep": "p_event", "p_trace": 0.4, "p_event": p_event, "floor": floor, **s_, "baseline_coverage": s_["levels"]["L1"]})
    for fl in [0.85, 0.9, 0.95]:
        a = run_spine(txs, recs, 0.4, p_proto=0.5, p_event=0.5, p_collision=0.05, floor=fl, calib=calib)
        s_ = summarize(recs, a)
        rows.append({"sweep": "floor", "p_trace": 0.4, "p_event": 0.5, "floor": fl, **s_, "baseline_coverage": s_["levels"]["L1"]})
    for r in rows:
        r.setdefault("floor", floor)
    return rows


# ----------------------------------------------------------------------------
# E2 reconciliation
# ----------------------------------------------------------------------------
def experiment_2(recs, capture=None):
    """Reconciliation on observable data only.
    The engine sees: (a) the attribution ledger (metered records: SKU, time, token quantities,
    rate-card estimate); (b) a FOCUS-style billing export: per SKU per day, billed quantity
    (rounded by the provider) and billed cost; (c) the settled price sheet; (d) the commitment
    contract for the provisioned pool. It never sees the planted effects directly.
    Planted into the bill: unmetered (shadow) usage; boundary-timing shifts; a provisioned-pool
    commitment with idle; one SKU overbilled on price; one SKU with omitted (under-billed) usage."""
    skus = list(SKUS)
    days = int(PERIOD_HOURS // 24)
    results = []
    for shadow in [0.0, 0.01, 0.02, 0.04, 0.08]:
        delta = {sk: float(rng.uniform(-0.02, 0.02)) for sk in skus}       # rate-card drift vs settled
        pool_sku, idle_frac = "B-frontier", 0.15
        over_sku, over_pct = "C-frontier", 0.03
        under_sku, under_pct = "B-mid", 0.02
        per = {}
        planted_unmetered_cost, planted_under_cost = {}, {}
        for sk in skus:
            ri, ro = SKUS[sk]
            m = [r for r in recs if r["sku"] == sk]
            day = np.array([min(int(r["t"] // 24), days - 1) for r in m])
            qin = np.array([r["tin"] for r in m], dtype=float); qout = np.array([r["tout"] for r in m], dtype=float)
            # --- ledger view (metered only), priced at the rate card
            led_in = np.bincount(day, weights=qin, minlength=days); led_out = np.bincount(day, weights=qout, minlength=days)
            ledger_est = ((led_in * ri + led_out * ro) / 1e6) * (1 + delta[sk])
            ledger_settled_cost = (led_in * ri + led_out * ro) / 1e6
            # --- true usage = metered + shadow (unmetered calls the ledger never saw)
            sh_in = led_in * shadow; sh_out = led_out * shadow
            true_in = led_in + sh_in; true_out = led_out + sh_out
            planted_unmetered_cost[sk] = float(((sh_in * ri + sh_out * ro) / 1e6).sum())
            # --- provider bill: boundary shift (last 6h of the period -> next period), rounding, errors
            last6 = np.array([r["t"] > PERIOD_HOURS - 6 for r in m])
            shift_sel = last6 & (rng.random(len(m)) < 0.5)
            shifted_in = qin[shift_sel].sum() * (1 + shadow); shifted_out = qout[shift_sel].sum() * (1 + shadow)
            bill_in = true_in.copy(); bill_out = true_out.copy()
            bill_in[-1] -= shifted_in; bill_out[-1] -= shifted_out                 # moved to next period day 1
            next_day1_in, next_day1_out = shifted_in, shifted_out
            if sk == under_sku:                                                     # provider omits usage
                planted_under_cost[sk] = -float(under_pct * ((bill_in * ri + bill_out * ro) / 1e6).sum())
                bill_in *= (1 - under_pct); bill_out *= (1 - under_pct)
            bill_in = np.round(bill_in / 1000) * 1000; bill_out = np.round(bill_out / 1000) * 1000   # provider rounding
            price_factor = (1 + over_pct) if sk == over_sku else 1.0
            billed_cost = ((bill_in * ri + bill_out * ro) / 1e6) * price_factor
            idle_commit = 0.0
            if sk == pool_sku:
                commit_cost = ledger_settled_cost.sum() / (1 - idle_frac)         # contract value, known to the engine
                idle_commit = commit_cost - ledger_settled_cost.sum()
                billed_cost = billed_cost + idle_commit / days                      # commitment billed evenly
            # ================= engine (observable data only) =================
            # (1) rate drift: restate ledger at the settled price sheet
            rate_drift = float((ledger_est - ledger_settled_cost).sum())
            # (2) price check (two-sided): implied billed unit price vs price sheet, using billed quantities
            bill_qty_cost_at_sheet = (bill_in * ri + bill_out * ro) / 1e6
            usage_bill = billed_cost - (idle_commit / days if sk == pool_sku else 0.0)   # commitment line is separate in the export
            with np.errstate(divide="ignore", invalid="ignore"):
                price_ratio = np.where(bill_qty_cost_at_sheet > 0, usage_bill / bill_qty_cost_at_sheet, 1.0)
            price_discrepancy = float(((price_ratio - 1.0) * bill_qty_cost_at_sheet).sum())
            # (3) timing: quantity missing on the last day that reappears on day 1 of the next period
            lastday_gap_in = led_in[-1] - bill_in[-1]; lastday_gap_out = led_out[-1] - bill_out[-1]
            timing_in = min(max(lastday_gap_in, 0.0), next_day1_in); timing_out = min(max(lastday_gap_out, 0.0), next_day1_out)
            timing_cost = float((timing_in * ri + timing_out * ro) / 1e6)
            # (4) idle: known from the commitment contract and the pool meter
            idle_adj = idle_commit
            # (5) residual: billed usage at sheet prices minus ledger at sheet prices, after timing
            residual = float(bill_qty_cost_at_sheet.sum() - ledger_settled_cost.sum() + timing_cost)
            per[sk] = {"billed": float(billed_cost.sum()), "rate_drift": rate_drift, "price_discrepancy": price_discrepancy,
                       "timing": timing_cost, "idle": idle_adj, "residual": residual}
            if capture is not None:  # record what this level's bill and inputs looked like (no RNG use)
                capture.setdefault(shadow, {})[sk] = {
                    "rate_card_drift": delta[sk],
                    "bill_in": bill_in.copy(), "bill_out": bill_out.copy(),
                    "commit_daily": idle_commit / days if sk == pool_sku else 0.0,
                    "billed_cost": billed_cost.copy(),
                    "next_in": float(next_day1_in), "next_out": float(next_day1_out),
                    "price_factor": price_factor,
                    "commitment_value": float(ledger_settled_cost.sum() + idle_commit) if sk == pool_sku else 0.0,
                    "pool_metered": float(ledger_settled_cost.sum()) if sk == pool_sku else 0.0,
                    "planted_unmetered": planted_unmetered_cost[sk],
                    "planted_under": planted_under_cost.get(sk, 0.0),
                    "over_pct": over_pct if sk == over_sku else 0.0,
                    "under_pct": under_pct if sk == under_sku else 0.0,
                    "timing_shift_in": float(shifted_in), "timing_shift_out": float(shifted_out),
                    "engine": dict(per[sk]),
                }
        tot_billed = sum(v["billed"] for v in per.values())
        clean = [sk for sk in skus if sk != under_sku]
        recovered = sum(per[sk]["residual"] for sk in clean)
        planted = sum(planted_unmetered_cost[sk] for sk in clean)
        results.append({
            "shadow_planted_pct": shadow * 100,
            "planted_pct_of_bill": 100 * planted / tot_billed,
            "recovered_pct_of_bill": 100 * recovered / tot_billed,
            "abs_error_pct_points": 100 * abs(recovered - planted) / tot_billed,
            "rate_drift_pct_of_bill": 100 * sum(v["rate_drift"] for v in per.values()) / tot_billed,
            "timing_pct_of_bill": 100 * sum(v["timing"] for v in per.values()) / tot_billed,
            "idle_pct_of_bill": 100 * sum(v["idle"] for v in per.values()) / tot_billed,
            "overbilling_flagged_usd": per[over_sku]["price_discrepancy"],
            "overbilling_planted_usd": over_pct * (per[over_sku]["billed"] / (1 + over_pct)),
            "underbilling_residual_usd": per[under_sku]["residual"] - planted_unmetered_cost[under_sku],
            "underbilling_planted_usd": planted_under_cost.get(under_sku, 0.0),
            "false_flags_other_skus_usd": sum(abs(per[sk]["price_discrepancy"]) for sk in skus if sk != over_sku),
        })
    return results


# ----------------------------------------------------------------------------
# E3 mid-flight gating
# ----------------------------------------------------------------------------
def trajectories(txs, recs):
    byid = {r["id"]: r for r in recs}
    out = []
    for tx in txs:
        rs = [byid[i] for i in tx["recs"]]
        cum = np.cumsum([r["cost"] for r in rs])
        out.append({"type": tx["type"], "runaway": tx["runaway"], "cum": cum, "final": float(cum[-1]),
                    "n": len(rs), "steps": [r["step"] for r in rs]})
    return out


def fit_forecaster(train):
    """Empirical conditional distribution of final cost given (type, stage, accumulated-spend quantile bin)."""
    model = {}
    # quantile bins of accumulated spend per (type, stage)
    for ttype in TEMPLATES:
        for stage in range(len(TEMPLATES[ttype])):
            acc, fin = [], []
            for tr in train:
                if tr["type"] != ttype:
                    continue
                # first index where this stage is reached
                idxs = [k for k, s in enumerate(tr["steps"]) if s == stage]
                if not idxs:
                    continue
                k = idxs[-1]  # after the last call of this stage (captures loops)
                acc.append(tr["cum"][k]); fin.append(tr["final"])
            if len(acc) < 50:
                continue
            acc, fin = np.array(acc), np.array(fin)
            qs = np.quantile(acc, [0.2, 0.4, 0.6, 0.8, 0.95])
            model[(ttype, stage)] = (qs, acc, fin)
    return model


def p_exceed(model, ttype, stage, acc_now, slo):
    key = (ttype, stage)
    if key not in model:
        return 0.0
    qs, acc, fin = model[key]
    b = int(np.searchsorted(qs, acc_now))
    edges = np.concatenate(([-np.inf], qs, [np.inf]))
    m = (acc >= edges[b]) & (acc < edges[b + 1])
    if m.sum() < 15:
        return float((fin > slo).mean())
    return float((fin[m] > slo).mean())


def experiment_3(train, test):
    model = fit_forecaster(train)
    rows = []
    slo = {t: float(np.quantile([tr["final"] for tr in train if tr["type"] == t and not tr["runaway"]], 0.95)) for t in TEMPLATES}
    base_finals = np.array([tr["final"] for tr in test])
    base_p95 = float(np.quantile(base_finals, 0.95))
    for theta in [None, 0.5, 0.7, 0.9, 0.95]:
        finals, gated, gated_nonrun, caught = [], 0, 0, 0
        n_nonrun = sum(1 for tr in test if not tr["runaway"]); n_run = len(test) - n_nonrun
        for tr in test:
            if theta is None:
                finals.append(tr["final"]); continue
            stopped = False
            for k, stage in enumerate(tr["steps"]):
                if k < 1:
                    continue
                if p_exceed(model, tr["type"], stage, tr["cum"][k], slo[tr["type"]]) > theta:
                    finals.append(float(tr["cum"][k]))  # stop after this call (approval hold)
                    stopped = True; gated += 1
                    if tr["runaway"]: caught += 1
                    else: gated_nonrun += 1
                    break
            if not stopped:
                finals.append(tr["final"])
        finals = np.array(finals)
        rows.append({
            "theta": theta if theta is not None else "ungated",
            "p95": float(np.quantile(finals, 0.95)), "p99": float(np.quantile(finals, 0.99)),
            "mean": float(finals.mean()), "spend_saved_pct": 100 * (1 - finals.sum() / base_finals.sum()),
            "false_positive_rate": gated_nonrun / n_nonrun, "runaway_caught_rate": (caught / n_run) if n_run else 0.0,
            "gated_share": gated / len(test),
        })
    return rows, base_p95


# ----------------------------------------------------------------------------
# Data export (deterministic: no random draws, so it cannot change any result)
# ----------------------------------------------------------------------------
def write_csv(path, header, rows):
    """Write a CSV; gzip it when the name ends in .gz, with a fixed timestamp so reruns are byte-identical."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    if path.endswith(".gz"):
        data = gzip.compress(data, compresslevel=9, mtime=0)
    with open(path, "wb") as f:
        f.write(data)


def export_sample(folder, txs, recs):
    """One generated sample: its business transactions and every model invocation, with ground truth."""
    cost_by_bt = {}
    for r in recs:
        cost_by_bt[r["bt"]] = cost_by_bt.get(r["bt"], 0.0) + r["cost"]
    write_csv(os.path.join(folder, "transactions.csv.gz"),
              ["bt_id", "bt_type", "tenant", "start_hour", "end_hour", "runaway", "n_invocations", "cost_usd"],
              ([tx["bt"], tx["type"], tx["tenant"], f"{tx['start']:.6f}", f"{tx['end']:.6f}", int(tx["runaway"]),
                len(tx["recs"]), f"{cost_by_bt.get(tx['bt'], 0.0):.8f}"] for tx in txs))
    write_csv(os.path.join(folder, "usage_records.csv.gz"),
              ["record_id", "true_bt_id", "bt_type", "tenant", "system", "agent", "step", "provider", "sku",
               "t_hour", "input_tokens", "output_tokens", "cost_usd", "failed", "coordination"],
              ([r["id"], r["bt"], r["type"], r["tenant"], r["system"], r["agent"], r["step"], PROVIDER_OF[r["sku"]],
                r["sku"], f"{r['t']:.6f}", r["tin"], r["tout"], f"{r['cost']:.8f}", int(r["failed"]), int(r["coord"])]
               for r in recs))


def f8(x):
    """Eight decimals, without printing tiny negatives as -0.00000000."""
    v = round(float(x), 8)
    return f"{(v or 0.0):.8f}"


def export_e2(folder, capture):
    """E2 inputs the reconciliation engine sees, the planted truth it never sees, and its decomposition."""
    days = int(PERIOD_HOURS // 24)
    write_csv(os.path.join(folder, "price_sheet.csv"),
              ["sku", "provider", "input_usd_per_mtok", "output_usd_per_mtok"],
              ([sk, PROVIDER_OF[sk], f"{ri:.4f}", f"{ro:.4f}"] for sk, (ri, ro) in SKUS.items()))
    for shadow, per in capture.items():
        sub = os.path.join(folder, f"shadow_{int(round(shadow * 100))}pct")
        rows = []
        for sk, c in per.items():
            ri, ro = SKUS[sk]
            for d in range(days):
                usage = float(c["billed_cost"][d]) - c["commit_daily"]
                rows.append(["current", d + 1, sk, PROVIDER_OF[sk], f"{c['bill_in'][d]:.0f}", f"{c['bill_out'][d]:.0f}",
                             f8(usage), f8(c['commit_daily']), f8(c['billed_cost'][d])])
            nxt = (c["next_in"] * ri + c["next_out"] * ro) / 1e6 * c["price_factor"]
            rows.append(["next", 1, sk, PROVIDER_OF[sk], f"{c['next_in']:.3f}", f"{c['next_out']:.3f}",
                         f8(nxt), f8(0.0), f8(nxt)])
        write_csv(os.path.join(sub, "billing_export.csv"),
                  ["period", "day", "sku", "provider", "billed_input_tokens", "billed_output_tokens",
                   "billed_usage_cost_usd", "commitment_cost_usd", "billed_cost_usd"], rows)
        write_csv(os.path.join(sub, "rate_card.csv"),
                  ["sku", "input_usd_per_mtok", "output_usd_per_mtok", "drift_vs_price_sheet"],
                  ([sk, f"{SKUS[sk][0] * (1 + c['rate_card_drift']):.6f}", f"{SKUS[sk][1] * (1 + c['rate_card_drift']):.6f}",
                    f"{c['rate_card_drift']:.6f}"] for sk, c in per.items()))
        write_csv(os.path.join(sub, "commitment_contract.csv"),
                  ["sku", "commitment_value_usd", "metered_consumption_usd", "idle_usd"],
                  ([sk, f"{c['commitment_value']:.6f}", f"{c['pool_metered']:.6f}",
                    f"{c['commitment_value'] - c['pool_metered']:.6f}"] for sk, c in per.items() if c["commitment_value"] > 0))
        write_csv(os.path.join(sub, "planted_ground_truth.csv"),
                  ["sku", "planted_unmetered_cost_usd", "overbilling_pct", "underbilling_pct", "planted_underbilling_usd",
                   "timing_shift_input_tokens", "timing_shift_output_tokens"],
                  ([sk, f8(c['planted_unmetered']), f"{100 * c['over_pct']:.2f}", f"{100 * c['under_pct']:.2f}",
                    f8(c['planted_under']), f"{c['timing_shift_in']:.3f}", f"{c['timing_shift_out']:.3f}"]
                   for sk, c in per.items()))
        write_csv(os.path.join(sub, "engine_decomposition.csv"),
                  ["sku", "billed_usd", "rate_drift_usd", "price_discrepancy_usd", "timing_usd", "idle_usd", "residual_usd"],
                  ([sk] + [f8(c['engine'][k]) for k in ("billed", "rate_drift", "price_discrepancy", "timing", "idle", "residual")]
                   for sk, c in per.items()))


# ----------------------------------------------------------------------------
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="Two-Ledger Model simulation study (synthetic data, fixed seed).")
    ap.add_argument("--out", default=OUT, help="folder for results, tables and figures (default: next to this script)")
    ap.add_argument("--export-data", metavar="DIR", default=None, help="also write every synthetic dataset to DIR as CSV")
    args = ap.parse_args(argv)
    OUT = os.path.abspath(args.out)
    os.makedirs(OUT, exist_ok=True)

    N = 20000
    txs, recs = gen_transactions(N)
    calib_txs, calib_recs = gen_transactions(4000)     # separate calibration sample (known truth)
    calib = calibrate_l4(calib_txs, calib_recs)
    e1 = experiment_1(txs, recs, calib)
    e2_capture = {} if args.export_data else None
    e2 = experiment_2(recs, capture=e2_capture)
    train_txs, train_recs = gen_transactions(12000)
    e3, base_p95 = experiment_3(trajectories(train_txs, train_recs), trajectories(txs, recs))
    if args.export_data:
        d = os.path.abspath(args.export_data)
        export_sample(os.path.join(d, "evaluation"), txs, recs)
        export_sample(os.path.join(d, "calibration"), calib_txs, calib_recs)
        export_sample(os.path.join(d, "training"), train_txs, train_recs)
        export_e2(os.path.join(d, "e2_reconciliation"), e2_capture)
        print(f"synthetic datasets written to {d}")
    res = {"seed": SEED, "n_transactions": N, "n_records": len(recs),
           "l4_calibration": {"score_edges": [float(x) if np.isfinite(x) else None for x in calib[0]], "precision": calib[1].tolist()},
           "E1": e1, "E2": e2, "E3": e3}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    # ---- LaTeX tables
    with open(os.path.join(OUT, "table_e1.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}ccccccccc@{}}\n\\toprule\n")
        f.write("$p_{\\mathrm{trace}}$ & $p_{\\mathrm{event}}$ & Floor & Trace-only cov. & Spine cost cov. & Cost acc. & Pooled & L4 share & Misattrib. \\\\\n\\midrule\n")
        for r in e1:
            f.write(f"{r['p_trace']:.1f} & {r['p_event']:.2f} & {r['floor']:.2f} & {100*r['baseline_coverage']:.1f}\\% & {100*r['cost_coverage']:.1f}\\% & {100*r['cost_accuracy']:.1f}\\% & {100*r['pooled_share']:.1f}\\% & {100*r['levels']['L4']:.1f}\\% & {100*r['mis_cost_share']:.2f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "table_e2.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\n")
        f.write("Planted unmetered & Recovered residual & Error (pp) & Rate drift & Timing & Overbilling flagged / planted & Under-billing residual / planted \\\\\n\\midrule\n")
        for r in e2:
            f.write(f"{r['planted_pct_of_bill']:.2f}\\% & {r['recovered_pct_of_bill']:.2f}\\% & {r['abs_error_pct_points']:.2f} & {r['rate_drift_pct_of_bill']:+.2f}\\% & {r['timing_pct_of_bill']:.2f}\\% & \\${r['overbilling_flagged_usd']:.2f} / \\${r['overbilling_planted_usd']:.2f} & \\${r['underbilling_residual_usd']:.2f} / \\${r['underbilling_planted_usd']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(OUT, "table_e3.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\n")
        f.write("Gate threshold $\\theta$ & p95 cost & p99 cost & Mean cost & Spend saved & False-positive rate & Runaways caught \\\\\n\\midrule\n")
        for r in e3:
            th = r['theta'] if isinstance(r['theta'], str) else f"{r['theta']:g}"  # :g keeps 0.95 as 0.95 (".1f" printed it as 0.9)
            f.write(f"{th} & \\${r['p95']:.3f} & \\${r['p99']:.3f} & \\${r['mean']:.4f} & {r['spend_saved_pct']:.1f}\\% & {100*r['false_positive_rate']:.2f}\\% & {100*r['runaway_caught_rate']:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    e1t = [r for r in e1 if r["sweep"] == "p_trace"]
    x = [r["p_trace"] for r in e1t]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(x, [100 * r["baseline_coverage"] for r in e1t], "k--", marker="s", label="Trace-only coverage")
    ax.plot(x, [100 * r["cost_coverage"] for r in e1t], "k-", marker="o", label="Spine cost coverage")
    ax.plot(x, [100 * r["cost_accuracy"] for r in e1t], "k:", marker="^", label="Spine cost accuracy")
    ax.set_xlabel("Share of invocations with trace context, $p_{\\mathrm{trace}}$"); ax.set_ylabel("Percent")
    ax.set_ylim(0, 102); ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_e1.pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "fig_e1.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot([r["planted_pct_of_bill"] for r in e2], [r["recovered_pct_of_bill"] for r in e2], "k-", marker="o", label="Recovered residual")
    ax.plot([0, max(r["planted_pct_of_bill"] for r in e2)], [0, max(r["planted_pct_of_bill"] for r in e2)], "k--", alpha=0.5, label="Perfect recovery")
    ax.set_xlabel("Planted unmetered spend (% of bill)"); ax.set_ylabel("Residual (% of bill)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_e2.pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "fig_e2.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    labels = [str(r["theta"]) for r in e3]
    ax.bar(labels, [r["p95"] for r in e3], color="0.35", label="p95 cost per transaction")
    ax2 = ax.twinx()
    ax2.plot(labels, [100 * r["false_positive_rate"] for r in e3], "k--", marker="o", label="False-positive rate (%)")
    ax.set_xlabel("Gate threshold $\\theta$"); ax.set_ylabel("p95 cost (USD)"); ax2.set_ylabel("False-positive rate (%)")
    ax.set_ylim(0, 0.36); ax2.set_ylim(0, 6.5)  # headroom so the legend sits above the bars and the FPR line
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", framealpha=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_e3.pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "fig_e3.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- console summary
    print(f"records={len(recs)} transactions={N}")
    print("L4 calibration precision by score decile:", np.round(calib[1], 3).tolist())
    for r in e1:
        print(f"E1 [{r['sweep']}] p_trace={r['p_trace']:.1f} p_event={r['p_event']:.2f} floor={r['floor']:.2f} trace-only={100*r['baseline_coverage']:.1f}% cov={100*r['coverage']:.1f}% costcov={100*r['cost_coverage']:.1f}% acc={100*r['accuracy']:.1f}% pooled={100*r['pooled_share']:.1f}% mis={100*r['mis_cost_share']:.2f}%")
    for r in e2:
        print(f"E2 planted={r['planted_pct_of_bill']:.2f}% recovered={r['recovered_pct_of_bill']:.2f}% err={r['abs_error_pct_points']:.2f}pp drift={r['rate_drift_pct_of_bill']:+.2f}% timing={r['timing_pct_of_bill']:.2f}% idle={r['idle_pct_of_bill']:.2f}% over flagged=${r['overbilling_flagged_usd']:.2f}/${r['overbilling_planted_usd']:.2f} under resid=${r['underbilling_residual_usd']:.2f}/planted=${r['underbilling_planted_usd']:.2f} falseflags=${r['false_flags_other_skus_usd']:.2f}")
    print(f"E3 base p95={base_p95:.4f}")
    for r in e3:
        print(f"E3 theta={r['theta']} p95={r['p95']:.4f} p99={r['p99']:.4f} mean={r['mean']:.5f} saved={r['spend_saved_pct']:.1f}% FPR={100*r['false_positive_rate']:.2f}% caught={100*r['runaway_caught_rate']:.1f}% gated={100*r['gated_share']:.1f}%")


if __name__ == "__main__":
    main()
