"""
Shared data loading + the single source of truth for how a forecast case is turned into
scorable bins.  Every downstream script (score / reliability / maps) iterates the SAME
`cases(...)` generator, so the resolution / censoring rules live in exactly one place.

See README §"Handling incomplete information" for the rationale behind every rule below.
"""
import numpy as np, pandas as pd, glob, os, importlib.util
from pathlib import Path
import config as C

# ── observed onsets ──────────────────────────────────────────────────────────
def load_onsets():
    """One row per subdistrict: id, onset_date. onset_date is YYYYMMDD (a column with any
    blank/no-onset rows reads as float -> '20260617.0') or an ISO date string; blank = no
    onset -> NaT."""
    df = pd.read_csv(C.ONSET_CSV())
    s = df["onset_date"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")            # compact YYYYMMDD
    iso = (dt.isna() & s.notna() & (s.str.len() == 10)).fillna(False)   # fallback: ISO 2026-06-17
    if iso.any():
        dt = dt.mask(iso, pd.to_datetime(s.where(iso), errors="coerce"))
    return {int(i): (o if pd.notna(o) else pd.NaT) for i, o in zip(df["id"], dt)}

def cutoff():
    """Date up to which a bin's outcome is knowable: a 5-day wet spell needs 5 observed days,
    so the last onset start day with a fully-observed window is OBS_END − 4."""
    return C.OBS_END - pd.Timedelta(days=4)

def dissemination_ids():
    return set(pd.read_csv(C.DISSEM_CSV())["id"].astype(int))

# ── unconditional climatology: 6-bin PMF for any init date F ─────────────────
class Climatology:
    """Wraps the wide climatology table: one row per subdistrict, one column per calendar
    day giving the climatological probability that onset falls on that day, plus a `later`
    column for mass beyond the tabulated window. `pmf6` re-bins this fixed daily PMF into
    the 6 categories relative to an arbitrary forecast init date F."""
    def __init__(self):
        u = pd.read_csv(C.CLIM_WIDE()); u["id"] = u["id"].astype(int); u = u.set_index("id")
        self.dcols = [c for c in u.columns if c != "later"]          # date columns (e.g. "06-04")
        self.dates = pd.to_datetime(["2026-" + c for c in self.dcols])
        self.arr   = u[self.dcols].to_numpy(float)                   # (n_subdistrict, n_days) daily onset prob
        self.later = u["later"].to_numpy(float) if "later" in u.columns else np.zeros(len(u))
        self.pos   = {i: k for k, i in enumerate(u.index)}           # subdistrict id -> row index
        self.tot   = self.arr.sum(1) + self.later                   # total mass (may be <1 -> deficit -> "later")
        self.index = u.index
    def pmf6(self, i, F):
        """6-bin unconditional-climatology PMF for subdistrict i at init F:
        [earlier(≤F), week1, week2, week3, week4, later(>F+28)]. Built by masking the daily
        PMF into date ranges relative to F and summing each range's probability."""
        k = self.pos[i]; a = self.arr[k]; d = self.dates
        me = d <= F                                                  # "earlier": days on/before init
        ml = d > F + pd.Timedelta(days=28)                           # tabulated days beyond week 4
        wk = [(d > F + pd.Timedelta(days=7*j)) & (d <= F + pd.Timedelta(days=7*(j+1))) for j in range(4)]  # week1..4 masks
        # "later" = tabulated post-week-4 mass + the explicit `later` column + any missing mass (1 - total)
        later = a[ml].sum() + self.later[k] + max(0.0, 1 - self.tot[k])
        return np.array([a[me].sum()] + [a[m].sum() for m in wk] + [later])
    def median_onset(self):
        """Climatological median onset date per subdistrict = first day the cumulative daily
        PMF reaches 0.5 (NaT if >50% of the mass falls past the tabulated window)."""
        cum = self.arr.cumsum(axis=1); out = {}
        for k, i in enumerate(self.index):
            j = np.searchsorted(cum[k], 0.5)                         # first index where CDF ≥ 0.5
            out[int(i)] = self.dates[j] if j < len(self.dates) else pd.NaT
        return out

# ── forecast registry + per-file loading (forecast PMF + conditional clim PMF)─
def forecast_registry():
    """Map every available forecast to its file: {(source, init_date): path}. Walks the
    forecast tree for dated folders named YYYYMMDDT00 and parses the date from the name.
    Uses blend_output_full.csv, or else the folder's other full-PMF blend file (e.g. the
    1 Jul issuance is named blend_output_jul1.csv)."""
    reg = {}
    for s in C.SOURCES:
        for d in sorted(glob.glob(str(C.FORECAST_ROOT / s / "**" / "2026*T00"), recursive=True)):
            n = os.path.basename(d)                                  # e.g. "20260610T00"
            files = [Path(d) / "blend_output_full.csv"] + sorted(Path(d).glob("blend_output_jul*.csv"))
            hit = next((f for f in files if f.exists()), None)
            if hit: reg[(s, pd.Timestamp(f"{n[0:4]}-{n[4:6]}-{n[6:8]}"))] = hit
    return reg

def _sig(x): return 1 / (1 + np.exp(-x))    # logit -> probability

def load_forecast(path):
    """Read one forecast file and return (ids, forecast PMF, conditional-clim PMF), each a
    5-column array over week1..later. The forecast PMF is stored directly; the conditional
    climatology is stored as logits for weeks 1-4, so we sigmoid them, put the leftover
    probability in 'later', and renormalize to sum to 1."""
    df = pd.read_csv(path); df["id"] = df["id"].astype(int)
    fc = df[C.WK_COLS].to_numpy(float)                                  # 5-bin forecast (wk1..later)
    cc = _sig(df[C.CLW_COLS].to_numpy(float))                           # P(week1..4) from logits
    cl = np.clip(1 - cc.sum(1, keepdims=True), 0, None)                 # remainder -> "later"
    clim = np.hstack([cc, cl]); clim = clim / clim.sum(1, keepdims=True)  # 5-bin conditional clim, normalized
    return df["id"].to_numpy(), fc, clim

def true_cat(o, F):
    """Which of the 6 categories the observed onset o falls in, relative to init date F.
    0 earlier(≤F) · 1 week1(≤F+7) · 2 week2 · 3 week3 · 4 week4(≤F+28) · 5 later(>F+28)."""
    dl = (o - F).days
    return 0 if dl <= 0 else 1 if dl <= 7 else 2 if dl <= 14 else 3 if dl <= 21 else 4 if dl <= 28 else 5

# ── THE CORE: iterate scorable cases ─────────────────────────────────────────
def cases(jobs=None, ids_subset=None):
    """
    Yield one dict per (subdistrict, forecast-date) case, with 6-bin PMFs for the three
    systems and the resolution / outcome bookkeeping.  Forecasts issued after a
    subdistrict's onset are skipped (the 'earlier' outcome would be trivial).

    Yielded dict keys:
      id, F                      subdistrict id, forecast init date
      p    {'fc','clim','uncond'} 6-bin PMFs (bin 0 = 'earlier')
      resolved  bool[6]          whether each bin's 0/1 outcome is knowable by cutoff
      oo        int[6]           the 0/1 outcome per bin (meaningful where resolved)
      true_bin  int | None       determined outcome category (None if not yet determinable)
      earlier   bool             True = onset already occurred at init -> dropped (see below)
      rps  {'fc','clim','uncond'}| None   RPS (sum of sq. cumulative errors over resolved thresholds)

    Cases dropped as `earlier` are yielded with ONLY {id, F, earlier: True} so callers can
    count them; all other keys are absent. Every non-earlier case has the full set of keys.
    """
    jobs = jobs if jobs is not None else C.FORECAST_JOBS
    cut = cutoff()                                        # last date whose outcome is knowable
    od = load_onsets()                                    # id -> onset date
    clim = Climatology()
    reg = forecast_registry()
    keep = None if ids_subset is None else set(ids_subset)  # optional id filter (e.g. dissemination)

    for s, F in jobs:                                     # each (source, init date) in the forecast set
        ids, fc, cr = load_forecast(reg[(s, F)])
        Ek  = [F + pd.Timedelta(days=7*k) for k in (1, 2, 3, 4)]   # week-end dates F+7 .. F+28
        F28 = F + pd.Timedelta(days=28)                  # end of week 4 (boundary of "later")
        ends = [F] + Ek                                  # cumulative-threshold end dates: F, F+7 .. F+28
        for r, i in enumerate(ids):
            if keep is not None and i not in keep: continue
            if i not in clim.pos: continue               # no climatology for this id -> can't score
            o = od.get(i, pd.NaT); known = pd.notna(o)   # onset observed?
            if known and o <= F:                              # onset already occurred at/before init
                yield dict(id=int(i), F=F, earlier=True)      # "earlier": dropped from scoring, but counted
                continue
            tc = true_cat(o, F) if known else None            # true category (None if onset not yet observed)
            # 6-bin PMFs. fc/clim have no 'earlier' probability (bin 0 = 0); uncond may.
            p = {"fc": np.array([0.0, *fc[r]]),
                 "clim": np.array([0.0, *cr[r]]),
                 "uncond": clim.pmf6(i, F)}
            # For each bin decide (a) is its 0/1 outcome KNOWABLE by the cutoff (`resolved`),
            # and (b) what that outcome IS (`oo`). A week is knowable if onset is already
            # observed, OR that week has fully elapsed by the cutoff (so "no onset" is certain).
            resolved = np.zeros(6, bool); oo = np.zeros(6, int)
            for k in range(6):
                if k == 0:
                    # 'earlier' is EXCLUDED from Brier/AUC (degenerate: outcome always 0 in the
                    # scored set, forecast/clim assign 0). It is STILL scored in RPS as a negative
                    # via the cumulative threshold at F (see the rps loop below).
                    resolved[k] = False; oo[k] = 0
                elif k <= 4:                                  # weeks 1-4
                    resolved[k] = known or (Ek[k-1] <= cut)          # onset seen, or week elapsed
                    oo[k] = 1 if (known and tc == k) else 0            # hit only if onset landed in this week
                else:                                         # k == 5, "later"
                    resolved[k] = known or (F28 <= cut)             # onset seen, or 4 weeks elapsed w/o onset
                    oo[k] = 1 if ((known and o > F28) or ((not known) and F28 <= cut)) else 0
            # Determined outcome category, used ONLY by coverage (which needs the exact true bin):
            # a known onset gives its category; an un-onset case is "later" once 4 weeks elapse;
            # otherwise the outcome isn't pinned down yet -> None -> excluded from coverage.
            if known:              true_bin = tc
            elif F28 <= cut:    true_bin = 5
            else:                  true_bin = None
            # RPS: sum of squared errors between the forecast CDF and the observed step CDF, over
            # the cumulative thresholds at ends = [F, F+7, .., F+28] that are resolved. The k=0
            # threshold (P[onset ≤ F]) is always included (this is where bin 0 IS scored).
            rps = {}
            for key in ("fc", "clim", "uncond"):
                cf = 0.0; se = []                                 # cf = running forecast CDF
                for k in range(5):                                # thresholds at ends[0..4] = F .. F+28
                    cf += p[key][k]                               # P(onset ≤ ends[k]) so far
                    thr_res = True if k == 0 else (known or (Ek[k-1] <= cut))   # is this threshold knowable?
                    if not thr_res: continue
                    ocdf = 1 if (known and o <= ends[k]) else 0   # observed CDF: 1 once onset has occurred
                    se.append((cf - ocdf) ** 2)
                rps[key] = float(np.sum(se)) if se else None      # summed over resolved thresholds
            yield dict(id=int(i), F=F, earlier=False, p=p, resolved=resolved, oo=oo,
                       true_bin=true_bin, rps=rps)
