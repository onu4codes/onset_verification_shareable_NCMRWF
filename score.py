"""
Score the forecast against conditional climatology (clim_raw, the "statistical model") and
unconditional climatology (uncond, "static climatology") on:
  • Brier      multinomial convention: per-bin mean × 5; lower better. Bin 0 "earlier" is
               EXCLUDED (always a negative in the scored set), so the pool is the 5 meaningful
               bins week1..later.
  • RPS        sum of squared cumulative errors over resolved thresholds; lower better. The
               threshold at F (P[onset<=F]) IS included, penalizing premature probability mass.
  • AUC        pooled one-hot (micro-averaged) discrimination; higher better. Bin 0 excluded.
  • coverage   fraction of determined cases whose true bin is in the prediction set:
      – coverage_1bin       set = single highest-probability bin
      – coverage_2contig    set = contiguous bin-pair with highest summed probability
      – coverage_adaptive   set = the window communicated in the message (MoAFW rule):
                            smallest k in 1..3 whose best contiguous k-bin window has summed
                            probability > 65%; if no such window, an "uncertain" message is
                            sent and the case is excluded.

Skill scores (BSS, RPSS = 1 − score/ref) are reported vs each reference. AUC and coverage are
reported as a difference (forecast − ref), since they have no natural ratio skill score.

Outputs (in output/<version>/, e.g. output/v1p1/ or output/v2/; suffixed _dissem for --dissemination):
  forecast_scores.csv          overall metrics (the headline table)
  forecast_scores_by_week.csv  N, positives, Brier, BSS and AUC separately for each lead bin
  verification_cases.csv       one row per scored (subdistrict, issuance) case: the exact
                               inputs to every metric above (probabilities, observed onset,
                               communicated window, hit/miss)

Usage:
    python score.py --version 1p1        # AIFS v1.1 blend forecasts -> output/v1p1/
    python score.py --version v2         # AIFS v2 blend forecasts   -> output/v2/
    python score.py --version v2 --dissemination   # restrict to the dissemination region
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import rankdata
import config as C, common

KEYS = ("fc", "clim", "uncond")                     # forecast, conditional clim, unconditional clim
BIN_NAMES = ["earlier", "week1", "week2", "week3", "week4", "later"]

def auc(p, o):
    """Area under the ROC curve via the Mann-Whitney rank identity: the probability that a
    random positive (o=1) is ranked above a random negative (o=0). NaN if all-one/all-zero."""
    p = np.asarray(p); o = np.asarray(o)
    if o.sum() == 0 or o.sum() == len(o): return np.nan
    r = rankdata(p); n1 = o.sum()
    return (r[o == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(o) - n1))

def prediction_sets(p6):
    """Fixed-size prediction sets for the "1 week" and "2 week" coverage rules:
    the single most-likely bin, and the highest-probability contiguous bin-pair."""
    top1 = {int(np.argmax(p6))}
    j = int(np.argmax([p6[k] + p6[k+1] for k in range(5)]))     # start of the best contiguous pair
    return top1, {j, j + 1}

def best_contig(p6, k):
    """Best contiguous window of size k: return (set of bin indices, summed probability)."""
    j = int(np.argmax([p6[s:s+k].sum() for s in range(len(p6) - k + 1)]))   # window start maximizing summed prob
    return set(range(j, j + k)), float(p6[j:j+k].sum())

def forecast_k(p6, thresh=C.COVERAGE_THRESH, kmax=C.COVERAGE_MAX_K):
    """The message rule: the smallest contiguous-set size 1..kmax whose summed probability
    exceeds `thresh` (65%). None => even kmax bins fall short => "uncertain" message."""
    for k in range(1, kmax + 1):
        if best_contig(p6, k)[1] > thresh:
            return k
    return None

def window_dates(bins, F):
    """Calendar dates of a contiguous set of bins relative to init F, e.g. '2026-06-04..2026-06-17'
    (or '>2026-07-08' if the window ends in the open-ended 'later' bin)."""
    lo, hi = min(bins), max(bins)
    start = F + pd.Timedelta(days=7 * (lo - 1) + 1)
    end = F + pd.Timedelta(days=7 * min(hi, 4))
    if lo == 0 and hi == 5: return "any"
    if lo == 0: return f"<={end:%Y-%m-%d}"
    if hi == 5: return f">{start - pd.Timedelta(days=1):%Y-%m-%d}"
    return f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"

def collect(jobs=None, ids_subset=None):
    """Iterate every scorable case and accumulate everything the metrics need.
      • Brier/AUC pool INDIVIDUAL resolved bins across all cases (P[k] vs O), also kept per bin.
      • RPS and coverage are PER-CASE, then averaged.
    Returns (results dict, per-case DataFrame)."""
    P = {k: [] for k in KEYS}; O = []; B = []             # pooled resolved bins, outcomes, bin index
    RPS = {k: [] for k in KEYS}                           # one RPS value per case
    cov1 = {k: [] for k in KEYS}; cov2 = {k: [] for k in KEYS}   # per-case hits: "1 week" / "2 week"
    covA = {k: [] for k in KEYS}; probA = {k: [] for k in KEYS}  # message rule: hits + window probability
    ksizes = []; n_uncertain = 0; n_total = 0; n_earlier = 0
    od = common.load_onsets(); rows = []
    for c in common.cases(jobs, ids_subset):
        n_total += 1
        if c["earlier"]:                                  # onset already occurred at init -> dropped, counted
            n_earlier += 1; continue
        m = c["resolved"]                                 # bool[6]; bin 0 is False (excluded)
        for k in KEYS: P[k].extend(c["p"][k][m])
        O.extend(c["oo"][m]); B.extend(np.flatnonzero(m))
        for k in KEYS:
            if c["rps"][k] is not None: RPS[k].append(c["rps"][k])
        o = od.get(c["id"], pd.NaT)
        row = dict(id=c["id"], init=c["F"].date(), onset_date=o.date() if pd.notna(o) else None,
                   true_bin=BIN_NAMES[c["true_bin"]] if c["true_bin"] is not None else None)
        for k in KEYS:
            row.update({f"p_{k}_{BIN_NAMES[b]}": round(float(c["p"][k][b]), 4) for b in range(1, 6)})
        # Coverage: only for cases whose true bin is fully determined
        if c["true_bin"] is not None:
            tb = c["true_bin"]
            for k in KEYS:
                s1, s2 = prediction_sets(c["p"][k])
                cov1[k].append(int(tb in s1)); cov2[k].append(int(tb in s2))
            kk = forecast_k(c["p"]["fc"])                 # the FORECAST sets the window size k
            row["msg_k"] = kk
            if kk is None:
                n_uncertain += 1; row["msg_window_fc"] = "uncertain"
            else:
                ksizes.append(kk)
                for key in KEYS:                          # each system uses ITS OWN best k-contiguous set
                    s, pr = best_contig(c["p"][key], kk)
                    covA[key].append(int(tb in s)); probA[key].append(pr)
                    row[f"msg_window_{key}"] = window_dates(s, c["F"])
                    row[f"msg_prob_{key}"] = round(pr, 4); row[f"msg_hit_{key}"] = int(tb in s)
        rows.append(row)
    O = np.array(O); B = np.array(B)
    out = {}
    for k in KEYS:
        p = np.array(P[k])
        out[k] = dict(
            Brier = C.BRIER_MULT * np.mean((p - O) ** 2),
            RPS   = float(np.mean(RPS[k])),
            AUC   = auc(p, O),
            coverage_1bin     = float(np.mean(cov1[k])),
            coverage_2contig  = float(np.mean(cov2[k])),
            coverage_adaptive = float(np.mean(covA[k])),
            msg_prob          = float(np.mean(probA[k])),
        )
        out[k]["by_bin"] = {b: dict(brier=np.mean((p[B == b] - O[B == b]) ** 2), auc=auc(p[B == b], O[B == b]))
                            for b in range(1, 6)}
    out["_by_bin_n"] = {b: (int((B == b).sum()), int(O[B == b].sum())) for b in range(1, 6)}
    n_det = len(cov1["fc"])
    out["_meta"] = dict(resolved_bins=len(O), positives=int(O.sum()),
                        rps_cases=len(RPS["fc"]), coverage_cases=n_det,
                        confident_cases=len(ksizes), uncertain_cases=n_uncertain,
                        uncertain_frac=n_uncertain / n_det if n_det else np.nan,
                        mean_set_size=float(np.mean(ksizes)) if ksizes else np.nan,
                        total_cases=n_total, earlier_cases=n_earlier,
                        earlier_frac=n_earlier / n_total if n_total else np.nan)
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c == "msg_k" or c.startswith("msg_hit_")]:
        df[col] = df[col].astype("Int64")                 # integer columns with blanks for uncertain cases
    return out, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(C.FORECAST_SETS),
                    help="AIFS version to score (see config.FORECAST_SETS)")
    ap.add_argument("--dissemination", action="store_true", help="restrict to the dissemination region")
    a = ap.parse_args()
    ids = common.dissemination_ids() if a.dissemination else None
    r, cases_df = collect(C.FORECAST_SETS[a.version], ids_subset=ids)
    tag = "_dissem" if a.dissemination else ""
    out_dir = C.OUTPUT_DIR / C.VERSION_DIR[a.version]
    mm = r["_meta"]
    print(f"forecast version: {a.version}  cutoff={common.cutoff():%Y-%m-%d}  "
          f"resolved bins={mm['resolved_bins']}  positives={mm['positives']}  "
          f"coverage cases={mm['coverage_cases']}")
    print(f"cases DROPPED (onset on/before init F): "
          f"{mm['earlier_cases']} ({mm['earlier_frac']:.1%} of {mm['total_cases']} forecast cases)")
    print(f"message rule (>{C.COVERAGE_THRESH:.0%}, up to {C.COVERAGE_MAX_K} contiguous bins): "
          f"{mm['confident_cases']} windows communicated, {mm['uncertain_frac']:.1%} 'uncertain' (excluded); "
          f"mean window {mm['mean_set_size']:.2f} weeks")
    # display names ("coverage_adaptive" = the communicated-window rule)
    DISPLAY = {"coverage_1bin": "cov_1week", "coverage_2contig": "cov_2week",
               "coverage_adaptive": "cov_msg_interval", "msg_prob": "msg_prob_stated"}
    print(f"{'metric':18}{'forecast':>11}{'clim_raw':>11}{'uncond':>11}{'skill/diff vs clim':>20}{'skill/diff vs uncond':>22}")
    rows = []
    for mt in ["Brier", "RPS", "AUC", "coverage_1bin", "coverage_2contig", "coverage_adaptive", "msg_prob"]:
        fc, cl, un = r["fc"][mt], r["clim"][mt], r["uncond"][mt]
        if mt in ("Brier", "RPS"):
            sc, su = 1 - fc/cl, 1 - fc/un                  # skill score 1 - score/ref (>0 beats ref)
        else:
            sc, su = fc - cl, fc - un                      # difference (AUC, coverage)
        name = DISPLAY.get(mt, mt)
        print(f"{name:18}{fc:>11.4f}{cl:>11.4f}{un:>11.4f}{sc:>+20.4f}{su:>+22.4f}")
        rows.append(dict(metric=name, forecast=round(fc, 4), clim_raw=round(cl, 4), uncond=round(un, 4),
                         vs_clim=round(sc, 4), vs_uncond=round(su, 4),
                         vs_clim_kind=("skill" if mt in ("Brier", "RPS") else "difference")))
    rows.append(dict(metric="uncertain_frac",       forecast=round(mm["uncertain_frac"], 4)))
    rows.append(dict(metric="mean_set_size",        forecast=round(mm["mean_set_size"], 4)))
    rows.append(dict(metric="dropped_onset_before_init_frac", forecast=round(mm["earlier_frac"], 4)))

    # per-lead-bin breakdown (Brier here is the plain per-bin mean squared error)
    print(f"\n{'bin':8}{'N':>8}{'pos':>7}{'Brier_fc':>10}{'BSS_clim':>10}{'BSS_unc':>10}{'AUC_fc':>8}{'AUC_clim':>9}{'AUC_unc':>8}")
    wk = []
    for b in range(1, 6):
        n, npos = r["_by_bin_n"][b]; bf, bc, bu = (r[k]["by_bin"][b]["brier"] for k in KEYS)
        af, ac, au = (r[k]["by_bin"][b]["auc"] for k in KEYS)
        print(f"{BIN_NAMES[b]:8}{n:>8}{npos:>7}{bf:>10.4f}{1-bf/bc:>+10.4f}{1-bf/bu:>+10.4f}{af:>8.3f}{ac:>9.3f}{au:>8.3f}")
        wk.append(dict(bin=BIN_NAMES[b], N=n, positives=npos, brier_fc=round(bf, 4), brier_clim=round(bc, 4),
                       brier_uncond=round(bu, 4), bss_clim=round(1-bf/bc, 4), bss_uncond=round(1-bf/bu, 4),
                       auc_fc=round(af, 4), auc_clim=round(ac, 4), auc_uncond=round(au, 4)))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("forecast_scores", pd.DataFrame(rows)), ("forecast_scores_by_week", pd.DataFrame(wk)),
                     ("verification_cases", cases_df)]:
        out = out_dir / f"{name}{tag}.csv"; df.to_csv(out, index=False); print("wrote", out)

if __name__ == "__main__":
    main()
