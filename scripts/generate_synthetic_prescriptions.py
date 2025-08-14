#!/usr/bin/env python3

import os, sys
from pathlib import Path

# add project root to sys.path BEFORE importing from utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json, os, io, pandas as pd, zipfile, tempfile
from utils.synth_prescriptions import generate_dataset
from utils.pdf_utils import text_to_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n","--num", type=int, default=200)
    ap.add_argument("--min-meds", type=int, default=1)
    ap.add_argument("--max-meds", type=int, default=3)
    ap.add_argument("--typo", type=float, default=0.2)
    ap.add_argument("--ocr", type=float, default=0.2)
    ap.add_argument("-o","--outdir", default="cleaned/synth_prescriptions")
    ap.add_argument("--pdf", action="store_true", help="also export each sample as a PDF")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds = generate_dataset(
        n=args.num,
        min_meds=args.min_meds,
        max_meds=args.max_meds,
        noise={"typo":args.typo, "ocr":args.ocr},
        seed=42
    )

    # Write JSONL + CSV
    jsonl_path = os.path.join(args.outdir, "dataset.jsonl")
    csv_path   = os.path.join(args.outdir, "dataset.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame([{"text": r["text"], "ground_truth": json.dumps(r["ground_truth"])} for r in ds])\
      .to_csv(csv_path, index=False)

    print(f"✅ Wrote {len(ds)} samples:")
    print(f"   - {jsonl_path}")
    print(f"   - {csv_path}")

    if args.pdf:
        pdf_dir = os.path.join(args.outdir, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        for i, r in enumerate(ds, start=1):
            text_to_pdf(r["text"], os.path.join(pdf_dir, f"rx_{i:04d}.pdf"))
        # zip convenience
        zipbuf = io.BytesIO()
        with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(pdf_dir)):
                if fn.lower().endswith(".pdf"):
                    z.write(os.path.join(pdf_dir, fn), fn)
        with open(os.path.join(args.outdir, "pdfs.zip"), "wb") as zf:
            zf.write(zipbuf.getvalue())
        print(f"   - {pdf_dir}/ (and {args.outdir}/pdfs.zip)")

if __name__ == "__main__":
    main()
