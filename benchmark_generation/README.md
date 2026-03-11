# Benchmark_generation

Generate Omniglot benchmark vectors in 128D and 256D using the same HDF5 layout as `coco-i2i-512-angular.hdf5`.

## Reference-Exact HDF5 Schema

Each output file `omniglot-<dim>-angular.hdf5` follows this schema:

- file attrs:
  - `dimension` (int)
  - `distance` = `angular`
  - `point_type` = `float`
  - `type` = `dense`
- datasets:
  - `train`: `float16`, shape `(N_train, dim)`
  - `test`: `float16`, shape `(N_test, dim)`
  - `neighbors`: `int64`, shape `(N_test, K)`
  - `distances`: `float64`, shape `(N_test, K)`

Notes:
- `train` comes from Omniglot background split.
- `test` comes from Omniglot evaluation split.
- `neighbors` and `distances` are cosine-based angular ground truth (`distance = 1 - cosine`).
- Default `K` is 100 to match the sample style.

## Script Features

- Trains an autoencoder and exports 128D/256D embeddings.
- Runs cosine `10-way-1-shot` evaluation every 10 epochs (default).
- Writes ANN benchmark style HDF5 files.

## Quick Start

```bash
python Benchmark_generation/generate_omniglot_benchmark_vectors.py \
  --dims 128 256 \
  --epochs 15 \
  --eval-every 10 \
  --neighbors-k 100
```

## Important Args

- `--neighbors-k`: top-k ground-truth neighbors per test vector (default `100`)
- `--gt-batch-size`: batch size for building neighbors/distances (default `256`)
- `--disable-normalize-embedding`: disable L2 normalization (not recommended for angular)

## Outputs

- HDF5 files in `Benchmark_generation/outputs/`
- `generation_summary.json` with training loss, few-shot history, and file metadata
