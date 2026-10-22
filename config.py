"""
Central configuration: input locations, the forecast set to score, and scoring constants.

All data lives under INPUT_ROOT = ./verification_data (self-contained, so the package is
shareable as one folder). Inputs are looked up with `resolve(...)`, which searches the tidy
per-type subfolders below; it also falls back to a flat layout, so pointing INPUT_ROOT at a
single directory containing all the files works too. Edit INPUT_ROOT to relocate the data.
"""
from pathlib import Path
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# 1. WHERE YOUR INPUTS LIVE.  Everything is under ./verification_data by default.
# ────────────────────────────────────────────────────────────────────────────
PKG_DIR    = Path(__file__).resolve().parent
INPUT_ROOT = PKG_DIR / "verification_data"       # <-- EDIT ME to relocate the data folder
OUTPUT_DIR = PKG_DIR / "output"

def resolve(*candidates, glob=None):
    """Return the first existing path among candidates, searched across INPUT_ROOT's
    per-type subfolders (and the root itself, for a flat layout). If glob is given, return
    the lexicographically-last match (i.e. the latest dated file)."""
    subdirs = ["", "onsets", "climatology", "support", "forecasts"]
    if glob:
        hits = []
        for s in subdirs:
            d = INPUT_ROOT / s
            if d.is_dir(): hits += sorted(d.glob(glob))
        if hits: return sorted(hits)[-1]
        raise FileNotFoundError(f"no file matching {glob!r} under {INPUT_ROOT}")
    for c in candidates:
        for s in subdirs:
            p = INPUT_ROOT / s / c
            if p.exists(): return p
    raise FileNotFoundError(f"none of {candidates} found under {INPUT_ROOT}")

# ────────────────────────────────────────────────────────────────────────────
# 2. INPUT FILES  (see README §"Input folders" for the verification_data/ layout)
# ────────────────────────────────────────────────────────────────────────────
FORECAST_ROOT = INPUT_ROOT / "forecasts"                 # holds the two "Blend output from ..." trees
ONSET_CSV     = lambda: resolve("onset_dates_2026.csv")  # one row per subdistrict: id, onset_date
CLIM_WIDE     = lambda: resolve("imd_subdistrict_esri_unc_clim_wide (1).csv",
                                "imd_subdistrict_esri_unc_clim_wide.csv")      # unconditional clim PMF
DISSEM_CSV    = lambda: resolve("dissemination_subdistricts.csv")              # dissemination region ids

# Forecast source-folder names inside FORECAST_ROOT
SRC_V11 = "Blend output from aifsv1.1"
SRC_V2  = "Blend output from aifsv2"
SOURCES = (SRC_V11, SRC_V2)

# ────────────────────────────────────────────────────────────────────────────
# 3. THE FORECAST SETS TO SCORE  (init dates are WEEKLY), one per AIFS version.
#    "1p1": the AIFS v1.1 blend, scored on its own, 13 May - 8 Jul.
#    "v2":  the AIFS v2 blend, scored on its own, 13 May - 8 Jul.
#    Each version is scored and written independently, into its own output subfolder
#    (see VERSION_DIR below). Each issuance covers only the subdistricts not yet
#    declared onset when it was issued.
# ────────────────────────────────────────────────────────────────────────────
T = pd.Timestamp
_WEEKLY_DATES = ["2026-05-13","2026-05-20","2026-05-27","2026-06-03","2026-06-10",
                  "2026-06-17","2026-06-24","2026-07-01","2026-07-08"]
FORECAST_SETS = {
    "1p1": [(SRC_V11, T(d)) for d in _WEEKLY_DATES],
    "v2":  [(SRC_V2,  T(d)) for d in _WEEKLY_DATES],
}
VERSION_DIR = {"1p1": "v1p1", "v2": "v2"}                   # --version value -> output subfolder name
FORECAST_JOBS = FORECAST_SETS["v2"]                         # fallback default (both versions are scored explicitly by run_all.py)

# ────────────────────────────────────────────────────────────────────────────
# 4. SCORING CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
# 6 ordered onset categories relative to each forecast init date F:
#   0 earlier (on/before F) | 1 week1 (F+1..F+7) | 2 week2 | 3 week3 | 4 week4 | 5 later (>F+28)
WK_COLS  = ["week1","week2","week3","week4","later"]         # forecast PMF columns
CLW_COLS = [f"prob_clim_mr_week{k}" for k in (1,2,3,4)]      # conditional-clim logit columns
BRIER_MULT = 5      # multinomial-Brier convention: per-bin mean × (#meaningful categories)
# Adaptive "coverage rate" rule: grow the contiguous prediction set 1->2->3 bins until its
# summed probability exceeds COVERAGE_THRESH; if even 3 bins fall short, the case is
# "uncertain" and excluded (the uncertain fraction is recorded).  The reference (climatology)
# is scored on ITS OWN best contiguous set of the SAME size the forecast settled on.
COVERAGE_THRESH = 0.65
COVERAGE_MAX_K  = 3
ONSET_SEASON_START = T("2026-06-04")                        # earliest onset start day considered
# Last day of rainfall data used to produce onset_dates_2026.csv. A week with no observed onset
# counts as "no onset" once it has elapsed by OBS_END - 4 days (a 5-day wet spell needs 5 days).
OBS_END = T("2026-09-16")
