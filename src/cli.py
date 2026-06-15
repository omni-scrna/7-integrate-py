"""Argument parsers for omnibenchmark 7-integrate-py modules."""

import argparse


def add_common_args(parser):
    """Args required by omnibenchmark for every module."""
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--name", type=str, required=True,
                        help="Module name/identifier")


def build_scanorama_parser():
    parser = argparse.ArgumentParser(description="OmniBenchmark integration module (scanorama)")
    add_common_args(parser)

    parser.add_argument("--pcas.tsv", dest="pcas_tsv", type=str, required=True,
                        help="TSV of uncorrected PCA embeddings (cell_id + PC columns)")
    parser.add_argument("--rawdata.h5ad", dest="rawdata_h5ad", type=str, required=True,
                        help="AnnData HDF5 file; obs is read for batch labels")
    parser.add_argument("--properties.info", dest="properties_info", type=str, required=True,
                        help="YAML file with batch_var field")
    parser.add_argument("--knn", type=int, required=True,
                        help="Number of nearest neighbors for batch alignment")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Bandwidth for Gaussian correction kernel")
    # accepted but unused
    parser.add_argument("--loadings.tsv", dest="loadings_tsv", default=None, help="Ignored")
    parser.add_argument("--normalized_selected.h5", dest="normalized_selected_h5",
                        default=None, help="Ignored")

    return parser
