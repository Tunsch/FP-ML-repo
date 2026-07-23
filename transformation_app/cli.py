#!/usr/bin/env python3
"""
cli.py

Kommandozeilen-Wrapper um core.py. Der Split ist standardmäßig AUS
(--apply-split zum Aktivieren) -- ohne Split bleibt in jeder Ausgabedatei
(außer combine=vector, dort steckt es im Dateinamen) die Spalte 'session'
erhalten, damit ihr den Split z.B. in einem Jupyter Notebook mit
core.train_test_split_by_session() selbst und später festlegen könnt.

Beispiele:
    # Nur Vektoren erzeugen, Split später im Notebook festlegen
    python3 cli.py sessions/ --outdir out --level 1 --combine label

    # Split direkt hier festlegen
    python3 cli.py sessions/ --outdir out --level 1 --combine label \
        --apply-split --test-ratio 0.2
"""

import argparse
import json
import sys
from pathlib import Path

from core import build_outputs, load_session, process_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_path", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("ei_samples"))
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--level", nargs="+", type=int, default=[1], choices=[1, 2, 3])
    ap.add_argument("--level2-mode", choices=["concat", "mean"], default="concat")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--impute-max-missing", type=int, default=0)
    ap.add_argument("--impute-max-gap", type=int, default=3)
    ap.add_argument("--combine", choices=["vector", "session", "label", "all"], default="vector")
    ap.add_argument("--apply-split", action="store_true",
                     help="Train/Test-Split bereits jetzt festlegen (Default: aus, 'session'-Spalte "
                          "bleibt für einen späteren Split z.B. im Notebook erhalten)")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--session-split", type=Path, default=None)
    args = ap.parse_args()

    if args.input_path.is_dir():
        pattern = f"**/{args.pattern}" if args.recursive else args.pattern
        csv_files = sorted(args.input_path.glob(pattern))
        if not csv_files:
            print(f"Keine Dateien passend zu '{args.pattern}' in {args.input_path} gefunden.", file=sys.stderr)
            sys.exit(1)
    else:
        csv_files = [args.input_path]

    session_split = json.loads(args.session_split.read_text()) if args.session_split else None

    per_file = []
    for csv_path in csv_files:
        print(f"--- {csv_path.name} ---")
        try:
            raw = load_session(csv_path, name_hint=csv_path.name)
        except ValueError as e:
            print(f"[Übersprungen] {e}", file=sys.stderr)
            continue
        res = process_file(csv_path.stem, raw, args.level, args.level2_mode,
                            log_transform=not args.no_log,
                            impute_max_missing=args.impute_max_missing,
                            impute_max_gap=args.impute_max_gap)
        for line in res["logs"]:
            print(" ", line.replace("\n", "\n  "))
        if not res["ok"]:
            print("  [Übersprungen] keine verwertbaren Zyklen.")
            continue
        per_file.append(res)

    if not per_file:
        print("Keine verwertbaren Daten gefunden.", file=sys.stderr)
        sys.exit(1)

    outputs = build_outputs(per_file, args.level, args.combine,
                             apply_split=args.apply_split, test_ratio=args.test_ratio,
                             seed=args.split_seed, session_split=session_split)

    for rel_path, content in outputs.items():
        path = args.outdir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    print(f"\nFertig. {len(outputs)} Datei(en) nach {args.outdir} geschrieben.")
    if not args.apply_split:
        print("Hinweis: kein Split angewendet (--apply-split nicht gesetzt). "
              "Die 'session'-Spalte (bzw. bei --combine vector der Dateiname) "
              "identifiziert die Ursprungs-Session für einen späteren Split.")


if __name__ == "__main__":
    main()
