"""Run the whole verification suite (scores + reliability diagrams) for one or both AIFS
forecast versions (v1.1, v2). Each version's outputs go to their own subfolder under output/
(output/v1p1/, output/v2/).
Usage: python run_all.py [--version 1p1|v2] [--dissemination]
       python run_all.py                      # runs both v1.1 and v2
       python run_all.py --version 1p1         # just the AIFS v1.1 blend
       python run_all.py --version v2          # just the AIFS v2 blend"""
import argparse, sys
import config as C, score, reliability

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(C.FORECAST_SETS), default=None,
                    help="run just this AIFS version; omit to run both")
    ap.add_argument("--dissemination", action="store_true")
    a = ap.parse_args()
    versions = [a.version] if a.version else list(C.FORECAST_SETS)
    for v in versions:
        print("=" * 70, f"\nSCORES: {v} forecasts\n", "=" * 70)
        sys.argv = ["score", "--version", v] + (["--dissemination"] if a.dissemination else []); score.main()
        print("\n" + "=" * 70, f"\nRELIABILITY: {v} forecasts\n", "=" * 70)
        sys.argv = ["reliability", "--version", v]; reliability.main()

if __name__ == "__main__":
    main()
