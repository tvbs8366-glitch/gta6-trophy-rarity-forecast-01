#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads every /input/<id>.in case (schema makefun/gta6-trophies-case/v1) and writes
/output/<id>.out: for each trophy, five quantiles (5/25/50/75/95%) of its
Exophase earn rate at the case's target. Standard library only, no network,
fully deterministic.

Model (median, in logit space) -- a blend of three views:
  A) Earners / players decomposition (weight 0.5).  A trophy's rate is
     earners / players-tracked.  Both counts grow, but on different clocks:
     new players dilute every rate at each Exophase refresh, while earners
     only grow as players actually earn the trophy.  We fit log(earners) and
     log(players) against log(1 + hours since release) on the recent distinct
     reads and extrapolate both to the target, so the forecast rate is
     earners(target) / players(target).  This is what makes the forecast move
     the right way when a flood of new players arrives.
  B) Direct logit trend (weight 0.3).  Least squares of logit(rate) against
     log(1 + hours since release) over the last 24 h of distinct reads, read
     at the target -- the house curve baseline's shape, refit on the data.
  C) Persistence (weight 0.2).  The trophy's last observed rate, which hedges
     the two extrapolations when growth saturates.

Fits use only *distinct* reads (consecutive reads with identical players and
rates are one stale Exophase page, not new information) and weight recent
reads more (24 h half-life), so the launch-night stale page cannot dominate.

Spread (per trophy, logit units) -- honest uncertainty, not a guess:
  sigma = sqrt(fit_residual^2 + (0.45*sqrt(horizon_days))^2 + refresh^2),
  where refresh is the typical logit jump seen at reads where players-tracked
  jumped (an Exophase refresh), or a wide prior when no refresh was ever
  observed and the page has been stale for hours (a refresh is then overdue
  and its dilution is unknown).  In that flat-and-stale case the median is
  also shifted down 0.6 logit: a refresh dilutes rates, and a page unchanged
  for hours is most likely to be refreshed next, not to stay frozen.

Quantiles are sigmoid(median_logit + z_tau * sigma), z the standard normal
quantiles, rounded to 4 decimals like the house baseline.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    return 100.0 / (1.0 + math.exp(-clamp(lg, -12.0, 12.0)))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b); b=0 if no spread."""
    sw = sum(ws)
    if sw <= 0 or not xs:
        return (ys[0] if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return my, 0.0
    b = sxy / sxx
    return my - b * mx, b


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are %r, not %r" % (case["levels"], LEVELS))
    if not case["history"]:
        raise ValueError("case has no history")

    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def x_of(moment):
        hours = (moment - RELEASE).total_seconds() / 3600.0
        return math.log1p(max(hours, 0.0))

    x_target = x_of(target)

    # Distinct reads only: drop a read identical to the previous kept one.
    distinct = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            distinct.append(read)
            prev_key = key
    if not distinct:
        distinct = list(case["history"])

    # Recency weights (24 h half-life) and a 48 h fit window when possible.
    def weight(moment):
        age_h = (checkpoint - moment).total_seconds() / 3600.0
        return 0.5 ** max(age_h, 0.0) / 24.0

    window = [r for r in distinct
              if (checkpoint - parse_time(r["at"])).total_seconds() <= 48 * 3600]
    if len(window) < 3:
        window = distinct
    recent24 = [r for r in distinct
                if (checkpoint - parse_time(r["at"])).total_seconds() <= 24 * 3600]
    if len(recent24) < 3:
        recent24 = window

    times_w = [parse_time(r["at"]) for r in window]
    xs_w = [x_of(t) for t in times_w]
    ws_w = [weight(t) for t in times_w]
    times_r = [parse_time(r["at"]) for r in recent24]
    xs_r = [x_of(t) for t in times_r]
    ws_r = [weight(t) for t in times_r]

    # --- players-tracked growth (shared by every trophy) ---
    log_p = [math.log(max(r["players_tracked"], 1)) for r in window]
    a_p, b_p = wols(xs_w, log_p, ws_w)
    log_p_last = log_p[-1]
    log_p_t = a_p + b_p * x_target
    log_p_t = log_p_last + clamp(log_p_t - log_p_last, -0.3, 2.0)

    # --- refresh jumps: logit moves at reads where players jumped ---
    jump_moves = []
    for j in range(1, len(distinct)):
        prev, cur = distinct[j - 1], distinct[j]
        if prev["players_tracked"] > 0 and \
                cur["players_tracked"] >= 1.03 * prev["players_tracked"]:
            for i in range(len(case["trophies"])):
                r0, r1 = prev["rates"][i], cur["rates"][i]
                if r0 is not None and r1 is not None:
                    jump_moves.append(abs(logit(r1) - logit(r0)))
    if jump_moves:
        sigma_refresh = clamp(0.7 * median(jump_moves), 0.0, 0.8)
    else:
        staleness_h = (checkpoint - parse_time(distinct[-1]["at"])).total_seconds() / 3600.0
        # No refresh ever observed; a page stale for hours is overdue one.
        sigma_refresh = 1.0 if staleness_h > 2.0 else 0.3

    # Flat-and-stale hazard: no movement ever seen, page long unchanged.
    flat_stale = len(distinct) <= 3 and \
        (checkpoint - parse_time(distinct[0]["at"])).total_seconds() / 3600.0 > 2.0

    sigma_rw = 0.45 * math.sqrt(max(horizon_days, 0.0))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last observed rate for this trophy.
        last_rate = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        if last_rate is None:
            last_rate = 1.0
        lg_last = logit(last_rate)

        # A) earners / players decomposition.
        xs_e, ys_e, ws_e = [], [], []
        for r, x, w in zip(window, xs_w, ws_w):
            rate = r["rates"][i]
            if rate is None:
                continue
            earners = max(rate / 100.0 * max(r["players_tracked"], 1), 0.5)
            xs_e.append(x)
            ys_e.append(math.log(earners))
            ws_e.append(w)
        if len(xs_e) >= 2:
            a_e, b_e = wols(xs_e, ys_e, ws_e)
            log_e_last = ys_e[-1]
            log_e_t = a_e + b_e * x_target
            log_e_t = log_e_last + clamp(log_e_t - log_e_last, -0.5, 2.5)
            rate_a = clamp(100.0 * math.exp(log_e_t - log_p_t), 0.01, 99.99)
        else:
            rate_a = last_rate
        lg_a = logit(rate_a)

        # B) direct logit trend over the last 24 h of distinct reads.
        xs_b, ys_b, ws_b = [], [], []
        for r, x, w in zip(recent24, xs_r, ws_r):
            rate = r["rates"][i]
            if rate is None:
                continue
            xs_b.append(x)
            ys_b.append(logit(rate))
            ws_b.append(w)
        if len(xs_b) >= 3:
            a_b, b_b = wols(xs_b, ys_b, ws_b)
            lg_b = a_b + b_b * x_target
            lg_b = lg_last + clamp(lg_b - lg_last, -3.0, 3.0)
            resid = [y - (a_b + b_b * x) for x, y in zip(xs_b, ys_b)]
            s_fit = clamp(math.sqrt(sum(v * v for v in resid) / len(resid)),
                          0.15, 0.8)
        else:
            lg_b = lg_last
            s_fit = 0.3

        # Median: blend, then the flat-and-stale refresh hazard shift.
        med = 0.5 * lg_a + 0.3 * lg_b + 0.2 * lg_last
        if flat_stale:
            med -= 0.6

        sigma = clamp(math.sqrt(s_fit ** 2 + sigma_rw ** 2 + sigma_refresh ** 2),
                      0.4, 1.8)
        qs = []
        for z in Z:
            q = round4(clamp(sigmoid(med + z * sigma), 0.01, 100.0))
            if qs and q < qs[-1]:
                q = qs[-1]
            qs.append(q)
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / ("%s.out" % path.stem)
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
