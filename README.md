# Monsoon onset forecast verification (2026)

Code and data to verify the 2026 subdistrict-level monsoon-onset forecasts: calibration of
the communicated onset windows, Brier Skill Score (BSS), AUC, RPS and reliability.
Everything needed is in this folder.

## How to run

```bash
pip install numpy pandas scipy matplotlib
python score.py --version 1p1      # AIFS v1.1 blend only: scores, per-week table, per-case dataset
python score.py --version v2       # AIFS v2 blend only
python run_all.py                  # both versions, plus reliability diagrams
python run_all.py --version v2     # just one version, plus its reliability diagrams
```

All available observations are used: a week counts as having no onset once it has fully
elapsed by `config.OBS_END` (16 Sep, the last day of rainfall data behind the onset file)
minus 4 days, since a 5-day wet spell needs 5 observed days. `--dissemination` restricts to the dissemination region; the forecast files already
cover only that region, so results are identical. Each version's outputs go to its own
subfolder under `output/` (`output/v1p1/`, `output/v2/`).

## Forecast versions

| Version | Option | What it is |
|---|---|---|
| **AIFS v1.1** | `--version 1p1` | The AIFS v1.1 blend, scored on its own: 9 weekly issuances, 13 May – 8 Jul. Each issuance covers only the subdistricts not yet declared onset when it was issued. |
| **AIFS v2** | `--version v2` | The AIFS v2 blend, scored on its own, same 9 weekly issuances. |

## Files

| File | Contents |
|---|---|
| `score.py` | Brier/BSS, RPS, AUC, coverage of communicated windows, stated message probability |
| `common.py` | data loading and the single definition of a scorable case (used by every script) |
| `config.py` | paths, the two forecast versions (`1p1`, `v2`), their output subfolders, the 65% message threshold |
| `reliability.py` | reliability diagrams per week and pooled |
| `run_all.py` | runs `score.py` and `reliability.py` for one or both forecast versions |

| Output (under `output/v1p1/` or `output/v2/`) | Contents |
|---|---|
| `forecast_scores.csv` | headline metrics for forecast, statistical model, static climatology |
| `forecast_scores_by_week.csv` | N, positives, Brier, BSS and AUC for each lead week |
| `verification_cases.csv` | one row per scored (subdistrict, issuance): probabilities, observed onset, communicated window, hit/miss — the dataset immediately before score calculation |
| `reliability_by_bin.png`, `reliability_joint.png` | reliability diagrams per week and with all weeks pooled |

(`_dissem` suffix on the CSVs = scored with `--dissemination`, restricting to the dissemination region.)

## Data (`verification_data/`)

| Path | Contents |
|---|---|
| `forecasts/Blend output from aifsv1.1/…/YYYYMMDDT00/blend_output_full.csv` | AIFS v1.1 blend forecasts, 13 May – 8 Jul |
| `forecasts/Blend output from aifsv2/…/YYYYMMDDT00/blend_output_full.csv` | AIFS v2 blend forecasts, 13 May – 8 Jul (1 Jul file is `blend_output_jul1.csv`) |
| `onsets/onset_dates_2026.csv` | observed 2026 onset date per subdistrict (blank = no onset) |
| `climatology/imd_subdistrict_esri_unc_clim_wide (1).csv` | static climatology: daily onset probabilities per subdistrict |
| `support/dissemination_subdistricts.csv` | ids of the subdistricts in the dissemination region |

Each forecast file gives, per subdistrict, the blended onset probabilities `week1…week4`,
`later` and the statistical model's weekly probabilities (`prob_clim_mr_week1…4`, stored as
logits).

Three systems are compared throughout: **forecast** (the blend), **statistical model**
(`clim_raw`, the conditional climatology in the forecast files) and **static climatology**
(`uncond`).

## Responses to the clarification questions

The answers to the verification clarification questions are in
[`Verification_clarifications_responses.docx`](Verification_clarifications_responses.docx).
They use the disseminated forecasts.
