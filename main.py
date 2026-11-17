"""gta6-twitch-ccu-forecast-01 entry: a launch-day shape forecast.

The forecast is built at run time from the case file's own comparables
(past big game launches' hourly Twitch counts), standard library only.
No network, no clock, no randomness: two runs write the same bytes.

The idea
--------
A launch day's hourly viewership, as a multiple of hour 1's print, is set
by WHEN the game unlocks around the world. Grand Theft Auto VI unlocks at
midnight local time, rolling: New Zealand first (11:00Z Nov 18), central
Europe 23:00Z, the UK 00:00Z, the US east coast 05:00Z and the west coast
08:00Z. So hour 1 (00:00-01:00Z) already has New Zealand, Australia, Asia
and Europe playing, the US joins from hour 6, and the count climbs into the
US evening. The comparables show three shapes, and only one matches:

  A  global unlock at ~00:00Z: hour 1 IS the peak, then a dip and a mild
     recovery (ELDEN RING, Cyberpunk 2077, ...). Hour 1 is 60-100% of peak.
  B  rolling midnight-local unlock: hour 1 is a real but partial audience,
     a dip in the small hours, then a climb to ~2x hour 1 by the evening
     (Call of Duty launches, Zelda, Monster Hunter Wilds, ...).
  C  one simultaneous unlock later in the day: hour 1 is the pre-launch
     trickle, then a 10x-1000x explosion (Silksong, Battlefield 6, ...).

GTA VI is shape B. The program therefore trains only on the comparables
whose own release day is shape B and clean (no early access, not weak):

  * drop the GTA VI target-history comparable and any comparable without a
    complete 24-hour release-day curve or with hour-1 viewers <= 0;
  * drop early-access-contaminated and weak tiers (parsed from the unlock
    text);
  * drop shape C (hour 1 under 10% of the day's peak) and shape A (the
    day's peak falls in the first 3 hours).

For each hour h = 2..24 it takes the median log-ratio
log(viewers(h) / viewers(1)) across those launches and the spread of the
log-ratios, and forecasts

    q_tau(h) = p1 * exp(median_h) * exp(z_tau * spread_h),

p1 the case's hour-1 print (1 if below 1), z_tau the standard normal's
quantile at 0.05/0.25/0.5/0.75/0.95. Two refinements, both read off the
comparables:

  * Hour-1 size conditions the evening ramp. Across the shape-B launches
    the evening ramp falls as hour 1 grows (correlation about -0.6 to -0.8
    at hours 18-24): a huge hour 1 means most of the world is already in,
    so there is less headroom. Hours 18-24 therefore use a linear
    regression of the log-ratio on log(hour-1 print) instead of the plain
    median, its slope shrunk by 0.7, and log(hour 1) clamped to the
    training range so a record-breaking hour 1 cannot extrapolate the line
    below a sensible ramp. Hours 15-17 blend the plain and the conditioned
    estimate so the shape has no step. The spread there is the regression's
    residual standard deviation.
  * The spread is floored at 0.15 and capped at 0.60 log units.

Local validation (leave-one-out over the 13 shape-B clean comparables in
the kit, scored with the rules' pinball loss on the log scale):
    persistence baseline (the starter)   0.330
    this model                           0.207
The starter scores about 0.387 on the practice cases; this model is built
to beat it on the final, where the comparables also include the four
practice launches (all shape B), which the program picks up automatically.

If anything about the comparables is unusable (too few survive, a missing
field, any error), the program falls back to the persistence baseline, so
it always writes a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The persistence baseline's spread (the starter's SIGMA), used as fallback.
BASELINE_SIGMA = 0.15

# --- model constants (chosen on the comparables; see the docstring) ---
# Hour 1 under this share of the day's peak means a late simultaneous
# unlock (shape C), not a rolling one.
CUTOFF_H1_PEAK = 0.10
# A day's peak in the first this many hours means a global 00:00Z unlock
# (shape A), not a rolling one.
A_PEAK_HOUR_MAX = 3
# Tiers to drop, matched (case-insensitively) in the unlock text.
DROP_TIERS = ("early-access-contaminated", "tier: weak", "target-history")
# Minimum shape-B clean comparables for the model; below this, fall back
# to all clean comparables, then to the baseline.
MIN_TRAIN = 5
# Hours from which the median is conditioned on the size of hour 1, and
# the first hour of the blend from the plain median to the conditioned one.
COND_FROM = 18
BLEND_FROM = 15
# The regression slope is shrunk by this much (0 = plain median, 1 = full).
SLOPE_SHRINK = 0.7
# Spread floor and cap, in natural-log units.
SPREAD_MIN = 0.15
SPREAD_MAX = 0.60
# A quantile is never written above this (the rules' cap).
Q_MAX = 1e9


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0). Same as the starter's.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def release_day_curve(comp: dict) -> list[float] | None:
    """The comparable's 24 hourly viewer counts for its release day, or None.

    A point at `from` covers that hour; the release day runs from
    day_start for 24 hours. Missing hours or a non-positive hour 1 make
    the comparable unusable (its ratios would be meaningless).
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    for pt in comp.get("hourly") or []:
        try:
            off = int((parse_time(pt["from"]) - day_start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= off < 24:
            by_hour[off] = float(pt["viewers"])
    if len(by_hour) < 24 or by_hour.get(0, 0) <= 0:
        return None
    return [by_hour[h] for h in range(24)]


def tier_drop(comp: dict) -> bool:
    """True when the comparable's tier says to leave it out."""
    name = str(comp.get("name", "")).lower()
    if "grand theft auto vi" in name:
        return True
    unlock = str(comp.get("unlock", "")).lower()
    return any(tag in unlock for tag in DROP_TIERS)


def shape_group(curve: list[float]) -> str:
    """A launch's release-day shape from its own curve: 'A', 'B' or 'C'."""
    h1 = curve[0]
    peak = max(curve)
    if h1 / peak < CUTOFF_H1_PEAK:
        return "C"
    if curve.index(peak) + 1 <= A_PEAK_HOUR_MAX:
        return "A"
    return "B"


def training_set(case: dict) -> list[list[float]]:
    """The shape-B clean comparables' release-day curves for this case."""
    clean: list[list[float]] = []
    shape_b: list[list[float]] = []
    for comp in case.get("comparables") or []:
        if tier_drop(comp):
            continue
        curve = release_day_curve(comp)
        if curve is None:
            continue
        clean.append(curve)
        if shape_group(curve) == "B":
            shape_b.append(curve)
    if len(shape_b) >= MIN_TRAIN:
        return shape_b
    if len(clean) >= MIN_TRAIN:
        return clean
    return []


def log_ratios(train: list[list[float]], h: int) -> list[float]:
    """log(viewers(h) / viewers(1)) across the training launches, h = 2..24."""
    return [math.log(max(1.0, c[h - 1]) / max(1.0, c[0])) for c in train]


def spread_of(xs: list[float]) -> float:
    """Sample standard deviation, floored and capped."""
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
    return min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(var)))


def conditioned(train: list[list[float]], h: int, p1: float) -> tuple[float, float]:
    """Median log-ratio and spread for hour h, conditioned on hour 1's size.

    A linear regression of the log-ratio on log(hour-1 print): bigger
    hour-1 prints ramp less in the evening. The slope is shrunk, and
    log(hour 1) is clamped to the training range, so a record-breaking
    hour 1 holds the ramp at the biggest launch seen, instead of
    extrapolating the line down. Returns (predicted log-ratio, spread),
    the spread being the regression's residual standard deviation.
    """
    lr = log_ratios(train, h)
    xs = [math.log(max(1.0, c[0])) for c in train]
    mx = sum(xs) / len(xs)
    my = sum(lr) / len(lr)
    vx = sum((x - mx) ** 2 for x in xs)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, lr))
    slope = (cov / vx if vx > 0 else 0.0) * SLOPE_SHRINK
    lx = math.log(max(1.0, p1))
    lx = min(max(lx, min(xs)), max(xs))
    pred = my + slope * (lx - mx)
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, lr)]
    var = sum(r * r for r in resid) / max(1, len(resid) - 1)
    return pred, min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(var)))


def hour_quantiles(p1: float, train: list[list[float]], h: int) -> list[float | int]:
    """The five quantiles of hour h's count, as multiples of hour 1's print."""
    lr = log_ratios(train, h)
    plain_med = median(lr)
    plain_spread = spread_of(lr)
    if h < BLEND_FROM:
        pred, spread = plain_med, plain_spread
    elif h >= COND_FROM:
        pred, spread = conditioned(train, h, p1)
    else:
        # Blend the plain median into the conditioned one over hours 15-17,
        # so the forecast shape has no step where the conditioning starts.
        w = (h - BLEND_FROM + 1) / (COND_FROM - BLEND_FROM + 1)
        cond_med, cond_spread = conditioned(train, h, p1)
        pred = (1.0 - w) * plain_med + w * cond_med
        spread = (1.0 - w) * plain_spread + w * cond_spread
    return [round2(min(Q_MAX, max(1.0, p1 * math.exp(pred) * math.exp(z * spread)))) for z in Z]


def baseline_quantiles(p1: float, h: int) -> list[float | int]:
    """The persistence baseline's quantiles (the starter's formula)."""
    return [round2(min(Q_MAX, max(1.0, p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        train = training_set(case)
    except Exception:
        train = []
    hours = []
    for row in case["forecast"]:
        h = int(row["hour"])
        q = hour_quantiles(p1, train, h) if train else baseline_quantiles(p1, h)
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
