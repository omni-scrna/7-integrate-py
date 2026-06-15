# 7-integrate-py

OmniBenchmark Python-based integration/batch-correction modules. Currently implements Scanorama (`scanorama`).

## Setup

```bash
pixi install
pixi run check
```

## Usage

```bash
python scanorama.py \
  --output_dir out/ \
  --name datasets \
  --pcas.tsv path/to/datasets_pcas.tsv \
  --rawdata.h5ad path/to/datasets.h5ad \
  --properties.info path/to/datasets_properties.yaml \
  --knn 20 \
  --sigma 15
```

## Output

TSV file `{output_dir}/{name}_corrected.tsv` with corrected embeddings in the same layout as the PCA input:

```
cell_id            corrected_dim1  corrected_dim2  ...  corrected_dim50
AAACCTGAGAAACCAT   0.312          -1.045           ...  0.008
```

## Entrypoints

| Entrypoint  | Script       | Method    |
|-------------|--------------|-----------|
| `scanorama` | scanorama.py | Scanorama |

## Parameters

| Parameter | Description                                    | Default |
|-----------|------------------------------------------------|---------|
| `knn`     | Number of nearest neighbors used to identify mutual nearest neighbors across batches | 20     |
| `sigma`   | Bandwidth for Gaussian correction kernel        | 15     |
