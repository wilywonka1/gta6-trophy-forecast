"""gta6-trophy-rarity-forecast-01 entry: refresh-aware counts forecaster.

The program reads one case file (schema makefun/gta6-trophies-case/v1):
every house read of the GTA VI Exophase trophy page before the checkpoint,
and writes, for each trophy, five quantiles of its earn rate (percent) at
the case target. Scoring is mean pinball loss on the logit scale.

Model, in one paragraph: Exophase refreshes in steps, so the history is
first compressed to its distinct page states (one per refresh epoch).
Players tracked and per-trophy earners (rate * players / 100) are then
forecast as counts, and rates are forecast directly, by an ensemble of
damped trend models whose slopes are shrunk toward the cross-trophy
median slope: persistence, logit-rate against log time, logit-rate
against log players, log earners against log players, and log earners
against log time, plus a recent-window (last 24h) log-time fit. The
per-trophy median of the ensemble (in logit space) is the forecast
median; a trust factor from internal backtests on earlier refresh
epochs scales how far the median may move from the last read. The
spread comes from those same backtest residuals, widened by horizon.
A history with a single distinct state is a stale page awaiting its
first refresh: the median is shifted down in logit (the refresh grows
the denominator far faster than earners) and the spread is widened.

Python 3.13, standard library only, deterministic, no network.
Run: python3 /bot/main.py with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
standing in for /input and /output outside the grader.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def pct_from_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    if text is None:
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def wlinfit(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


def shrink_slope(b, med, spread, npts):
    """Shrink a per-trophy slope toward the pooled median slope."""
    if b is None:
        return med
    w = npts / (npts + 3.0)
    out = med + (b - med) * w
    lo = med - max(1.25 * spread, 0.35)
    hi = med + max(1.25 * spread, 0.35)
    return min(hi, max(lo, out))


def distinct_states(history):
    """Compress consecutive identical page states (one per refresh)."""
    states = []
    seen_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == seen_key:
            continue
        seen_key = key
        t = parse_time(read.get("at") or read.get("slot"))
        states.append(
            {
                "t": t if t is not None else 0.0,
                "players": float(read.get("players_tracked") or 0),
                "rates": list(rates),
            }
        )
    return states


def forecast_players(states, x_last, x_target):
    """Forecast players tracked at the target; returns a count >= last."""
    p_last = states[-1]["players"]
    if p_last <= 0:
        return 1.0
    if len(states) < 2:
        return p_last
    xs = [s["x"] for s in states]
    ln_p = [math.log(max(1.0, s["players"])) for s in states]
    # Recent epochs weigh most: player growth decelerates, and the last
    # epochs carry the growth rate that continues into the horizon.
    ws = [0.3 + 1.7 * (i / max(1, len(states) - 1)) ** 2 for i in range(len(states))]
    preds = [ln_p[-1]]
    fit = wlinfit(xs, ln_p, ws)
    if fit is not None:
        b = min(2.5, max(0.0, fit[1])) * 0.8
        preds.append(ln_p[-1] + b * (x_target - x_last))
    # Level fit of players against log time, damped the same way.
    fit2 = wlinfit(xs, [s["players"] for s in states], ws)
    if fit2 is not None:
        slope = max(0.0, fit2[1]) * 0.8
        p_lin = states[-1]["players"] + slope * (x_target - x_last)
        if p_lin > 0:
            preds.append(math.log(p_lin))
    ln_target = median(preds)
    ln_target = min(ln_target, ln_p[-1] + 3.0)
    return max(p_last, math.exp(ln_target))


def core_forecast(states, target_age_h, n_trophies):
    """Ensemble median logit per trophy for the given target age (hours).

    `states` are distinct states carrying precomputed x = ln(1 + age_h)
    and age_h. Returns (median_logits, last_logits, p_target).
    """
    x_last = states[-1]["x"]
    x_target = math.log(1.0 + max(0.0, target_age_h))
    p_target = forecast_players(states, x_last, x_target)
    ln_p_target = math.log(max(1.0, p_target))
    n = len(states)

    # Per-trophy series over the distinct states.
    series = []
    for i in range(n_trophies):
        pts = []  # (x, ln_players, logit_rate, ln_earners)
        for s in states:
            rate = s["rates"][i] if i < len(s["rates"]) else None
            if rate is None:
                continue
            rate = float(rate)
            earn = rate * s["players"] / 100.0
            pts.append(
                (
                    s["x"],
                    math.log(max(1.0, s["players"])),
                    logit(rate),
                    math.log(earn + 0.5),
                )
            )
        series.append(pts)

    def slopes(idx_x, idx_y, recent_only=False):
        out = []
        for pts in series:
            use = pts
            if recent_only and pts:
                use = [p for p in pts if p[0] >= x_last - 1.1]
                if len(use) < 3:
                    use = pts
            if len(use) < 2:
                out.append(None)
                continue
            xs = [p[idx_x] for p in use]
            ys = [p[idx_y] for p in use]
            # Weight recent epochs more (the regime that carries into the
            # target) and precise epochs more: a rate printed to 2 decimals
            # from a small tracked base is quantised (a handful of earners
            # is a visible logit jump), while a large base pins it down.
            # pts carry ln(players) at index 1, so precision ~ sqrt(players).
            span = max(1e-9, x_last - xs[0])
            ws = []
            for p in use:
                recency = 0.3 + 1.7 * ((p[0] - xs[0]) / span) ** 2
                precision = min(3.0, max(0.35, math.exp(0.5 * (p[1] - use[-1][1]) + 1.0)))
                ws.append(recency * precision)
            fit = wlinfit(xs, ys, ws)
            out.append(fit[1] if fit else None)
        return out

    def pooled(raw):
        vals = sorted(v for v in raw if v is not None)
        if not vals:
            return 0.0, 1.0
        med = median(vals)
        q1 = quantile(vals, 0.25)
        q3 = quantile(vals, 0.75)
        return med, max(0.2, q3 - q1)

    raw_time = slopes(0, 2)
    raw_time_recent = slopes(0, 2, recent_only=True)
    raw_plog = slopes(1, 2)
    raw_earn_p = slopes(1, 3)
    raw_earn_t = slopes(0, 3)
    med_t, spr_t = pooled(raw_time)
    med_tr, spr_tr = pooled(raw_time_recent)
    med_p, spr_p = pooled(raw_plog)
    med_ep, spr_ep = pooled(raw_earn_p)
    med_et, spr_et = pooled(raw_earn_t)

    medians = []
    lasts = []
    for i, pts in enumerate(series):
        if not pts:
            medians.append(logit(0.5))
            lasts.append(logit(0.5))
            continue
        last_x, last_lnp, last_lg, last_lne = pts[-1]
        lasts.append(last_lg)
        preds = [last_lg]  # persistence
        dx_t = x_target - last_x
        if raw_time[i] is not None and n >= 2:
            b = shrink_slope(raw_time[i], med_t, spr_t, len(pts))
            d = min(3.0, max(-3.0, b * dx_t))
            preds.append(last_lg + d)
        if raw_time_recent[i] is not None and n >= 3:
            b = shrink_slope(raw_time_recent[i], med_tr, spr_tr, len(pts))
            d = min(3.0, max(-3.0, b * dx_t * 0.9))
            preds.append(last_lg + d)
        d_lnp = ln_p_target - last_lnp
        if raw_plog[i] is not None and n >= 2:
            b = shrink_slope(raw_plog[i], med_p, spr_p, len(pts))
            d = min(3.0, max(-3.0, b * d_lnp))
            preds.append(last_lg + d)
        if raw_earn_p[i] is not None and n >= 2:
            b = shrink_slope(raw_earn_p[i], med_ep, spr_ep, len(pts))
            b = min(1.7, max(-0.3, b))
            ln_e = last_lne + b * d_lnp
            earn = max(0.0, math.exp(ln_e) - 0.5)
            rate = min(100.0, 100.0 * earn / p_target) if p_target > 0 else 0.0
            preds.append(logit(rate))
        if raw_earn_t[i] is not None and n >= 2:
            b = shrink_slope(raw_earn_t[i], med_et, spr_et, len(pts))
            b = min(4.0, max(-0.5, b))
            ln_e = last_lne + b * dx_t
            earn = max(0.0, math.exp(ln_e) - 0.5)
            earn = max(earn, math.exp(last_lne) - 0.5)  # earners never shrink
            rate = min(100.0, 100.0 * earn / p_target) if p_target > 0 else 0.0
            preds.append(logit(rate))
        medians.append(median(preds))
    return medians, lasts, p_target


def backtest(states, n_trophies):
    """Fit on earlier refresh epochs, predict the last epoch.

    Returns (trust, sigma_per_sqrt_day) from ensemble logit errors.
    """
    n = len(states)
    if n < 5:
        return 1.0, None
    cut_fracs = [0.55, 0.75, 0.9]
    errs = []
    errs_persist = []
    for frac in cut_fracs:
        cut = max(3, int(round(n * frac)))
        if cut >= n:
            continue
        sub = states[:cut]
        target_age = states[-1]["age_h"]
        horizon_days = max(0.02, (states[-1]["t"] - sub[-1]["t"]) / 86400.0)
        meds, lasts, _ = core_forecast(sub, target_age, n_trophies)
        for i in range(n_trophies):
            actual = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
            if actual is None:
                continue
            y = logit(float(actual))
            errs.append((meds[i] - y) / math.sqrt(horizon_days))
            errs_persist.append((lasts[i] - y) / math.sqrt(horizon_days))
    if len(errs) < 20:
        return 1.0, None
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    rms_p = math.sqrt(sum(e * e for e in errs_persist) / len(errs_persist))
    if rms_p <= 1e-9:
        trust = 1.0
    else:
        trust = min(1.0, max(0.35, rms_p / (rms + 1e-9) * 0.85))
    return trust, rms


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_trophies = len(trophies)

    release = case.get("game", {}).get("release_date")
    anchor = None
    if release:
        try:
            anchor = parse_time(str(release) + "T00:00:00Z")
        except Exception:
            anchor = None
    first_t = parse_time(history[0].get("at") or history[0].get("slot")) or 0.0
    if anchor is None or anchor > first_t + 86400:
        anchor = first_t
    target_t = parse_time(case.get("target"))
    at_t = parse_time(case.get("at"))
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    if horizon_days <= 0 and target_t and at_t:
        horizon_days = max(0.01, (target_t - at_t) / 86400.0)

    states = distinct_states(history)
    for s in states:
        s["age_h"] = max(0.0, (s["t"] - anchor) / 3600.0)
        s["x"] = math.log(1.0 + s["age_h"])
    if target_t:
        target_age = max(0.0, (target_t - anchor) / 3600.0)
    else:
        target_age = states[-1]["age_h"] + horizon_days * 24.0

    last_read_t = parse_time(history[-1].get("at") or history[-1].get("slot")) or states[-1]["t"]
    span_h = max(0.0, (last_read_t - first_t) / 3600.0)

    meds, lasts, _p_target = core_forecast(states, target_age, n_trophies)
    trust, sigma_psd = backtest(states, n_trophies)

    if sigma_psd is None:
        sigma_psd = 0.55
    sigma_psd = min(1.1, max(0.3, sigma_psd))
    base_sigma = sigma_psd * math.sqrt(max(0.02, horizon_days))
    base_sigma = min(1.6, max(0.3, base_sigma))

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = states[-1]["rates"][i] if i < len(states[-1]["rates"]) else None
        last_rate = float(last_rate) if last_rate is not None else 0.5
        lg_last = lasts[i]
        lg_med = lg_last + trust * (meds[i] - lg_last)
        sigma = base_sigma
        # A trophy pinned at the boundary in every state stays there.
        all_rates = [
            s["rates"][i]
            for s in states
            if i < len(s["rates"]) and s["rates"][i] is not None
        ]
        pinned_high = all_rates and all(float(r) >= 99.99 for r in all_rates)
        pinned_low = all_rates and all(float(r) <= 0.01 for r in all_rates)
        if pinned_high:
            lg_med = logit(99.99)
            sigma = 0.18
        elif pinned_low:
            lg_med = logit(0.02)
            sigma = 0.45
        elif len(states) == 1:
            # Stale page, first refresh still to come: the denominator
            # jumps far faster than earners, so rates fall in logit and
            # the honest spread is wide whatever the horizon.
            if 0.05 < last_rate < 99.9:
                shift = 0.9 if span_h >= 1.0 else 0.45
                lg_med = lg_last - shift
            sigma = min(1.6, max(1.0, 0.85 + 0.3 * math.sqrt(max(0.0, horizon_days))))
        else:
            move = abs(lg_med - lg_last)
            sigma = base_sigma * (1.0 + 0.22 * min(1.5, move))
            earn_last = last_rate * states[-1]["players"] / 100.0
            sigma *= 1.0 + 0.35 / math.sqrt(earn_last + 1.0)
            sigma = min(1.9, sigma)
        qs = []
        for z in Z:
            q = pct_from_logit(lg_med + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
