"""Run the whole verification suite (scores + reliability diagrams) for one or all
(AIFS version x forecast tree) combinations. Each version's outputs go to their own
subfolder under output/ (output/v1p1/, output/v2/); the --dissemination value controls
which forecast tree is scored and adds a `_dissem` suffix to the dissemination-tree files
so both trees' outputs can live side by side in the same subfolder.

Usage:
    python run_all.py                                       # all 4 combinations
    python run_all.py --version v1p1                        # v1.1, both trees
    python run_all.py --dissemination True                   # both versions, Dissemination tree only
    python run_all.py --dissemination True  --version v1p1   # Dissemination outputs for v1p1
    python run_all.py --dissemination True  --version v2     # Dissemination outputs for v2
    python run_all.py --dissemination False --version v1p1   # All_weekly outputs for v1p1
    python run_all.py --dissemination False --version v2     # All_weekly outputs for v2
"""
import argparse, sys
import config as C, score, reliability, common

VERSIONS = ["v1p1", "v2"]          # canonical CLI spellings (common.VERSION_ALIASES maps these)
DISSEM_OPTIONS = [True, False]

def run_one(version, dissemination):
    tree = "Dissemination" if dissemination else "All_weekly"
    print("=" * 70, f"\nSCORES: {version} forecasts, {tree} tree\n", "=" * 70)
    sys.argv = ["score", "--version", version, "--dissemination", str(dissemination)]
    score.main()
    print("\n" + "=" * 70, f"\nRELIABILITY: {version} forecasts, {tree} tree\n", "=" * 70)
    sys.argv = ["reliability", "--version", version, "--dissemination", str(dissemination)]
    reliability.main()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=VERSIONS, default=None,
                    help="run just this AIFS version (v1p1 or v2); omit to run both")
    ap.add_argument("--dissemination", type=common.str2bool, nargs="?", const=True, default=None,
                    help="True = Dissemination tree only, False = All_weekly tree only; "
                         "omit to run both trees")
    a = ap.parse_args()
    versions = [a.version] if a.version else VERSIONS
    dissem_opts = [a.dissemination] if a.dissemination is not None else DISSEM_OPTIONS
    for v in versions:
        for d in dissem_opts:
            run_one(v, d)

if __name__ == "__main__":
    main()
