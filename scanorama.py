#!/usr/bin/env python3
# Remove the script's own directory from sys.path to avoid shadowing
# the installed 'scanorama' package with this file (scanorama.py).
import sys as _sys
import os as _os
import argparse
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_sys.path = [p for p in _sys.path if _os.path.abspath(p) != _this_dir and p != ""]

"""Scanorama integration module for omnibenchmark.

Reads uncorrected PCA embeddings (pcas.tsv) and batch labels from the obs
group of rawdata.h5ad, runs Scanorama in PCA space, and writes corrected
embeddings in the same TSV layout as the input.

Output
------
File: {output_dir}/{name}_corrected.tsv

Tab-separated, with header row:
  cell_id  corrected_dim1  corrected_dim2  ...

One row per cell; values are float64 corrected embedding coordinates.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import scanorama as scanorama_lib

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli
from writers import Embedding, write_embeddings  # noqa: E402


def parse_args():
    # src/common/cli injects the shared contract (base args + the
    # `INTG8` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="INTG8 module (scanpy-backed)")
    cli.add_base_args(p)               # --output_dir, --name
    cli.add_stage_args(p, "INTG8")  # --normalized_selected_h5  
    p.add_argument("--knn", type=int, required=True, help="Number of KNNs")
    p.add_argument("--sigma", type=int, required=True, help="sigma parameter")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "pcas_tsv", "rawdata_h5ad", "properties_info",
              "knn", "sigma"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # read properties.info to get batch variable
    import yaml
    with open(args.properties_info) as f:
        props = yaml.safe_load(f)
    batch_var = props.get("batch_var")
    if not batch_var:
        raise ValueError("batch_var is required in properties.info for integration")
    print(f"  batch_variable (from properties.info): {batch_var}")

    # read PCA embeddings
    df = pl.read_csv(args.pcas_tsv, separator="\t")
    row_ids = df["cell_id"].to_list()
    matrix = df.drop("cell_id").to_numpy().astype(np.float64)
    print(f"  embedding (cells x dims): {matrix.shape}")

    # read batch labels from rawdata.h5ad obs
    import h5py
    with h5py.File(args.rawdata_h5ad, "r") as h5:
        h5_cell_ids = [
            s.decode() if isinstance(s, bytes) else s
            for s in h5["obs/_index"][:]
        ]
        h5_batch_vals = [
            s.decode() if isinstance(s, bytes) else s
            for s in h5[f"obs/{batch_var}"][:]
        ]
    batch_map = dict(zip(h5_cell_ids, h5_batch_vals))

    # align batch labels to PCA cell order
    cell_batches = [batch_map[c] for c in row_ids]
    batch_order = list(dict.fromkeys(cell_batches))  # unique, preserving order
    print(f"  batch variable '{batch_var}': {len(batch_order)} levels "
          f"({', '.join(batch_order)})")

    # split embedding by batch; treat PC dimensions as "genes" so scanorama
    # aligns in PCA space
    batch_indices = {b: [i for i, c in enumerate(cell_batches) if c == b]
                     for b in batch_order}
    n_dims = matrix.shape[1]
    pc_names = [f"PC{i+1}" for i in range(n_dims)]
    embeddings = [matrix[batch_indices[b]] for b in batch_order]
    genes_per_batch = [pc_names for _ in batch_order]

    integrated, _ = scanorama_lib.integrate(
        embeddings, genes_per_batch,
        knn=args.knn,
        sigma=args.sigma,
    )

    # reassemble in original cell order
    corrected = np.empty((len(row_ids), integrated[0].shape[1]), dtype=np.float64)
    for idxs, b_integrated in zip(batch_indices.values(), integrated):
        corrected[idxs] = b_integrated
    print(f"  corrected embedding: {corrected.shape}")

    out_emb = Embedding(
        matrix=corrected,
        row_ids=row_ids,
        col_names=[f"corrected_dim{i + 1}" for i in range(corrected.shape[1])],
    )
    out = Path(args.output_dir) / f"{args.name}_corrected.tsv"
    write_embeddings(out_emb, out)
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
