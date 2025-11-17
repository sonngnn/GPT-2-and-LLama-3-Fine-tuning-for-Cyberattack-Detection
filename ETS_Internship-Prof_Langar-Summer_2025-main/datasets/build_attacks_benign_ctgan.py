#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_attacks_benign_ctgan.py

─────────────────────────────────────────────────
Goal
-----
Produce a final ~2,000,000-row CSV from the CIC IoT 2023 dataset by:
- Capping each attack class at CAP_ATTACK rows (balanced attacks),
- Then adding Benign rows to reach TOTAL_TARGET.

Why this design?
----------------
• The CIC IoT 2023 raw export is usually split across multiple files (part-*.csv).
  This script streams those files in chunks to control memory use and to sample
  rows per class efficiently.

• Labels in CIC IoT 2023 are not perfectly uniform (e.g., 'DDoS-*', 'BenignTraffic').
  We canonicalize labels so downstream balancing is reliable.

• Some attack classes may be under-represented. To avoid biasing the dataset,
  we train a CTGAN per minority attack class (CPU-friendly settings) to
  synthesize new rows and reach the cap. When too few real rows exist, we fall
  back to duplication (explicitly documented below).

• SDV 1.9 introduced sdv.single_table.CTGANSynthesizer with SingleTableMetadata.
  We prefer this path when available and fall back to the legacy sdv.tabular.CTGAN.
  Binary (0/1) columns are cast to boolean for correct metadata typing, then
  cast back to 0/1 at sampling time.

• We keep memory footprint modest:
  - Chunked reading (CHUNKSIZE),
  - Streaming into per-class temporary files,
  - Training one CTGAN per minority class only.

• Reproducibility: we fix Python, NumPy, and Torch seeds, and we always sample
  with a fixed RANDOM_SEED.

Updates
-------------
• SDV 1.9 path: sdv.single_table.CTGANSynthesizer (+ SingleTableMetadata).
• Robust counting by summation (no accidental overwrites).
• Low-memory streaming + per-class CTGAN for minority classes.
• Treat {0,1} columns as boolean during training; cast back to 0/1 on sampling.
• CPU-only profile: reduced CTGAN epochs (80) to keep runtime reasonable.
• Step [4/5] bug fix: Benign rows are actually written (no "available=0 → write 0" issue).

Output
------
CICIoT2023_attacks_benign_CTGAN.csv
with a single header row matching the original columns.

Input format assumptions
------------------------
• Directory contains multiple CSV chunks named "part-*.csv".
• Each CSV has a 'label' column.
• CSVs can be comma-separated; if parsing detects only one column, we auto-detect
  the delimiter (sep=None, python engine).
• We don't enforce numeric types; we use pandas defaults (optionally with pyarrow).

Edge cases & safeguards
-----------------------
• If 'label' is missing → we abort with a clear error.
• If no 'part-*.csv' are found → we abort with a clear error.
• If after canonicalization 'DDoS' is absent → we abort (indicates label issues).
• For very small classes (< MIN_GAN_ROWS), we duplicate instead of training a GAN.
• If Benign available rows < needed, we oversample (with replacement) to fill.

Performance notes
-----------------
• CTGAN (SDV single-table) trains with metadata inferred from the (boolean-
  corrected) DataFrame. For legacy CTGAN, we detect discrete columns (object or
  strict {0,1}) and pass them explicitly.

• We use CPU-only settings, pac/batch size are set on the legacy CTGAN path.
• tqdm is optional; if missing, we degrade gracefully.

"""

from __future__ import annotations
import csv
import glob
import random
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch

# ===================== USER SETTINGS =====================
# Directory expected to contain 'part-*.csv' files.
DATA_DIR     = Path("./")
# Output CSV path.
OUT_FILE     = Path("CICIoT2023_attacks_benign_CTGAN.csv")
# Target total number of rows in the final dataset.
TOTAL_TARGET = 2_000_000
# Upper cap per attack class (≈ 60% of 2M split across ~22 attack classes).
CAP_ATTACK   = 54_545
# Chunk size for streaming CSV reading (memory control).
CHUNKSIZE    = 200_000
# Global seed for reproducibility (Python/NumPy/Torch and sampling).
RANDOM_SEED  = 42
# If a class has fewer than this many real rows, skip CTGAN and duplicate instead.
MIN_GAN_ROWS = 100
# CPU-friendly number of epochs for CTGAN training.
CTGAN_EPOCHS = 80
# =========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Optional Pandas acceleration (pyarrow backend) when available.
if hasattr(pd.options, "mode") and hasattr(pd.options.mode, "dtype_backend"):
    try:
        pd.options.mode.dtype_backend = "pyarrow"
    except Exception:
        # If pyarrow isn't installed or not supported, proceed with defaults.
        pass

# Optional progress bars (tqdm). If unavailable, define a no-op wrapper.
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x, **k): return x

# SDV 1.9 (single_table) preferred; fall back to legacy sdv.tabular.CTGAN if needed.
try:
    from sdv.single_table import CTGANSynthesizer as CTGAN_ST
    from sdv.metadata import SingleTableMetadata
    SDV_SINGLE_TABLE = True
except Exception:
    SDV_SINGLE_TABLE = False
    try:
        # Legacy CTGAN (older SDV versions)
        from sdv.tabular import CTGANSynthesizer as CTGAN_LEGACY  # type: ignore
    except Exception as e:
        raise SystemExit(
            "\n[ERROR] Could not import SDV. Please install sdv>=1.9 (single_table) "
            "or a compatible legacy version.\n"
            f"Import error: {e}\n"
        )

# ============== Label canonicalization ===================
def canon_label(lbl: str) -> str:
    """
    Normalize label strings to reduce class fragmentation.
    - Any label starting with 'DDoS' → 'DDoS'
    - 'BenignTraffic' → 'Benign'
    - Otherwise, keep unchanged.

    Rationale:
    CIC IoT 2023 may include 'DDoS-XYZ' variants that we merge into a single
    'DDoS' bucket for balanced capping; Benign may appear as 'BenignTraffic'.
    """
    s = str(lbl).strip()
    s_low = s.lower()
    if s_low.startswith("ddos"):
        return "DDoS"
    if s_low == "benigntraffic":
        return "Benign"
    return s

# ============== Robust CSV chunk reader ==================
def smart_chunks(path: str, chunksize: int):
    """
    Read CSV in chunks with robust delimiter handling.

    Strategy:
    1) Try sep="," (fast path).
    2) If that yields a single-column frame (common sign of wrong sep),
       fall back to sep=None with the Python engine (auto-detection).
    3) If any exception occurs, retry with the auto-detect path.

    We set on_bad_lines="skip" to tolerate occasional malformed rows without
    stopping the stream.
    """
    try:
        for c in pd.read_csv(path, sep=",", chunksize=chunksize,
                             low_memory=False, on_bad_lines="skip"):
            if len(c.columns) == 1:  # likely delimiter mismatch
                for c2 in pd.read_csv(path, sep=None, engine="python",
                                      chunksize=chunksize, low_memory=False,
                                      on_bad_lines="skip"):
                    yield c2
                return
            yield c
    except Exception:
        for c2 in pd.read_csv(path, sep=None, engine="python",
                              chunksize=chunksize, low_memory=False,
                              on_bad_lines="skip"):
            yield c2

# ============== CTGAN helpers (SDV 1.9 path) =============
def detect_binary_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns that are strictly {0,1} (ignoring NaNs).
    These columns should be cast to boolean for correct metadata typing
    with SingleTableMetadata; we cast them back to 0/1 after sampling.
    """
    out = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.isin([0, 1]).all():
            out.append(c)
    return out

def fit_ctgan_sdv19(df_train: pd.DataFrame, epochs: int):
    """
    Prepare metadata and train CTGAN (SDV 1.9 single_table API).

    Steps:
    - Convert 0/1 columns to boolean so metadata infers 'boolean' type.
    - Auto-detect metadata from the DataFrame.
    - Fit CTGAN with a CPU-friendly epoch count.

    We store the list of binary columns on the synthesizer to recast
    booleans back to 0/1 after sampling (keeps parity with original schema).
    """
    bin_cols = detect_binary_columns(df_train)
    df_bool = df_train.copy()
    for c in bin_cols:
        df_bool[c] = df_bool[c].astype(bool)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_bool)

    synth = CTGAN_ST(
        metadata=metadata,
        epochs=epochs,
        verbose=True,   # show training progress
    )
    synth.fit(df_bool)
    # Remember binary columns for post-sample recast.
    synth._binary_cols = bin_cols  # type: ignore[attr-defined]
    return synth

def ctgan_sample_sdv19(synth, n: int) -> pd.DataFrame:
    """
    Sample n synthetic rows and recast boolean columns back to 0/1 ints.
    """
    df = synth.sample(n)
    for c in getattr(synth, "_binary_cols", []):
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df

# ================== Main pipeline ========================
def main():
    # Discover input split files.
    csv_files = sorted(glob.glob(str(DATA_DIR / "part-*.csv")))
    if not csv_files:
        raise SystemExit(f"No 'part-*.csv' found in {DATA_DIR.resolve()}")

    print(f"[DEBUG] {len(csv_files)} input files detected.")
    for n in csv_files[:3]:
        print("  ", Path(n).name)
    if len(csv_files) > 3:
        print("  …")

    # ---------- [1/5] Count rows per (canonical) label ----------
    print("\n[1/5] Counting flows per label…")
    counts = defaultdict(int)
    for f in tqdm(csv_files):
        for ch in smart_chunks(f, CHUNKSIZE):
            if "label" not in ch.columns:
                raise SystemExit(f"{f} does not contain a 'label' column")
            ch = ch.copy()
            ch["label"] = ch["label"].map(canon_label)
            vc = ch["label"].value_counts()
            for k, v in vc.items():
                counts[k] += int(v)

    print("\n[INFO] Raw counts (canonical labels):")
    total_avail = 0
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"   {k:<22} {v:>10,}")
        total_avail += v
    print(f"   {'TOTAL':<22} {total_avail:>10,}")

    attack_labels = [c for c in counts if c != "Benign"]
    if "DDoS" not in attack_labels:
        # If this triggers, the dataset label mapping likely changed.
        raise SystemExit("DDoS not found after canonicalization — please check label values.")

    # Target rows per attack class = min(actual DDoS count, CAP_ATTACK)
    # (We use DDoS as a reference upper bound to avoid setting an unachievable cap.)
    attack_target = min(counts["DDoS"], CAP_ATTACK)
    benign_needed_global = max(TOTAL_TARGET - attack_target * len(attack_labels), 0)

    print(f"\n[INFO] attack_target = {attack_target:,} per attack class")
    print(f"[INFO] benign_needed = {benign_needed_global:,} rows to reach {TOTAL_TARGET:,}")

    # ---------- [2/5] Stream input into per-class temp files ----------
    print("\n[2/5] Streaming parts into per-class temp files…")
    tmp_dir = Path(tempfile.mkdtemp(prefix="ciciot_ctgan_"))
    tmp_paths = {lab: tmp_dir / f"tmp_{lab}.csv" for lab in (attack_labels + ["Benign"])}

    writers: dict[str, csv.writer] = {}
    out_files = {}
    header = None

    # We cap attacks at min(count, attack_target). Benign is collected up to global need.
    caps = {lab: min(counts[lab], attack_target) for lab in attack_labels}
    caps["Benign"] = benign_needed_global

    kept = defaultdict(int)

    # Open one temp CSV per label (no headers; we store raw rows for speed).
    for lab, p in tmp_paths.items():
        fh = p.open("w", newline="")
        out_files[lab] = fh
        writers[lab] = csv.writer(fh)

    # Stream over all parts, sampling per label until each cap is met.
    stop_stream = False
    for f in tqdm(csv_files):
        if stop_stream:
            break
        for ch in smart_chunks(f, CHUNKSIZE):
            if header is None:
                header = ch.columns.tolist()
                if "label" not in header:
                    raise SystemExit("CSV without 'label' column.")
            ch = ch.copy()
            ch["label"] = ch["label"].map(canon_label)

            for lab, df_lab in ch.groupby("label", sort=False):
                if lab not in writers:
                    continue
                need_cap = caps[lab]
                if need_cap <= 0:
                    continue
                rem = need_cap - kept[lab]
                if rem <= 0:
                    continue
                # Sample without replacement within this chunk.
                take = df_lab if len(df_lab) <= rem else df_lab.sample(n=rem, random_state=RANDOM_SEED, replace=False)
                writers[lab].writerows(take.values.tolist())
                kept[lab] += len(take)

            # Stop early if all caps are satisfied.
            if all(kept[k] >= caps[k] for k in caps):
                stop_stream = True
                break

    for fh in out_files.values():
        fh.close()

    print("\n[INFO] Temp collected (original rows only):")
    for k in (attack_labels + ["Benign"]):
        print(f"   {k:<22} {kept[k]:>10,}")

    # ---------- [3/5] Build balanced attack classes ----------
    print("\n[3/5] Building balanced attack classes…")
    fout = OUT_FILE.open("w", newline="")
    writer_final = csv.writer(fout)
    writer_final.writerow(header)  # single header row in the final output

    total_written = 0

    for lab in attack_labels:
        path = tmp_paths[lab]
        df = pd.read_csv(path, header=None, names=header)
        rows_real = len(df)
        need = attack_target - rows_real

        # Exclude the label column from the CTGAN training set.
        df_train = df.drop(columns=["label"])

        if rows_real >= attack_target:
            # Down-sample if we collected more than the cap.
            print(f"   {lab:<22} real={rows_real:>7,}  → write {attack_target:>7,} (down-sample)")
            df_out = df.sample(n=attack_target, random_state=RANDOM_SEED, replace=False)
            writer_final.writerows(df_out.values.tolist())
            total_written += len(df_out)

        else:
            # Minority class: synthesize the missing rows or duplicate if too small.
            if rows_real >= MIN_GAN_ROWS:
                print(f"   {lab:<22} real={rows_real:>7,}  → CTGAN synth {need:>7,}")
                if SDV_SINGLE_TABLE:
                    synth = fit_ctgan_sdv19(df_train, epochs=CTGAN_EPOCHS)
                    synth_rows = ctgan_sample_sdv19(synth, need)
                else:
                    # Legacy SDV: detect discrete columns (object or strict {0,1})
                    discretes = []
                    for c in df_train.columns:
                        s = df_train[c].dropna()
                        if s.empty:
                            continue
                        if s.dtype == "object" or (pd.api.types.is_numeric_dtype(s) and s.isin([0, 1]).all()):
                            discretes.append(c)
                    synth = CTGAN_LEGACY(
                        epochs=CTGAN_EPOCHS,
                        batch_size=512,
                        pac=5,
                        cuda=False,             # CPU-only
                        verbose=True,
                        random_state=RANDOM_SEED,
                    )
                    synth.fit(df_train, discrete_columns=discretes)
                    synth_rows = synth.sample(need)

                # Re-attach the label for synthetic rows and concatenate.
                synth_rows["label"] = lab
                df_out = pd.concat([df, synth_rows], ignore_index=True)
                writer_final.writerows(df_out.values.tolist())
                total_written += len(df_out)
            else:
                # Too few real samples to train a meaningful CTGAN → duplication fallback.
                print(f"   {lab:<22} real={rows_real:>7,}  → DUPLICATE {need:>7,} (fallback)")
                extra = df.sample(n=need, random_state=RANDOM_SEED, replace=True)
                df_out = pd.concat([df, extra], ignore_index=True)
                writer_final.writerows(df_out.values.tolist())
                total_written += len(df_out)

        # Clean up temp file for this class.
        try:
            path.unlink()
        except Exception:
            pass

    # ---------- [4/5] Add Benign to reach TOTAL_TARGET ----------
    print("\n[4/5] Adding Benign to reach the global target…")
    benign_path = tmp_paths["Benign"]
    if benign_path.exists():
        ben = pd.read_csv(benign_path, header=None, names=header)
    else:
        ben = pd.DataFrame(columns=header)

    benign_needed_now = max(TOTAL_TARGET - total_written, 0)
    avail = len(ben)

    if benign_needed_now <= 0:
        print("   Nothing to add (already at target).")
    else:
        if avail >= benign_needed_now:
            print(f"   Benign available={avail:,}  → write {benign_needed_now:,}")
            df_benign_out = ben.sample(n=benign_needed_now, random_state=RANDOM_SEED, replace=False)
        else:
            need_extra = benign_needed_now - avail
            print(f"   Benign available={avail:,}  → write {benign_needed_now:,} (dup {need_extra:,})")
            extra = ben.sample(n=need_extra, random_state=RANDOM_SEED, replace=True) if need_extra > 0 else ben.iloc[0:0]
            df_benign_out = pd.concat([ben, extra], ignore_index=True)

        writer_final.writerows(df_benign_out.values.tolist())
        total_written += len(df_benign_out)

    fout.close()

    # ---------- [5/5] Cleanup ----------
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print(f"\n[5/5] DONE → {OUT_FILE.resolve()}")
    print(f"Total rows written: {total_written:,}")

if __name__ == "__main__":
    main()
