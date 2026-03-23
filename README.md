# Spectral Drug Verification

A proof of concept for instrument-based compound identification using Raman spectral data.

## Overview

This project explores the workflow structure behind spectral drug verification: taking measured spectral signals, preprocessing them, and using machine learning to classify compound identity.

The central idea is simple: if compounds have overlapping spectral fingerprints, the problem is not just “train a model”. It becomes a system problem involving signal quality, preprocessing, evaluation, reproducibility, and ultimately deployability.

This repository was built to explore that problem in a more professional, end-to-end way than a notebook-only analysis.


## Problem the project explores

This project is designed around a spectral drug verification problem:

- **identity classification** — determining which compound is present from its spectrum
- **scientific measurement data** — working with Raman spectra rather than standard tabular business data
- **workflow reliability** — structuring preprocessing, modelling, and evaluation in a reusable pipeline
- exposing a trained scientific ML workflow through a lightweight inference API

The current implementation focuses on classification first. Concentration estimation is a later extension.

## Data used

The raw dataset is stored in:

`data/raw/raman_spectra_api_compounds.csv`

It consists of:

- spectral columns from `150.0` to `3425.0`
- one target label column: `label`

This is a **real spectral dataset**, not a fully synthetic toy dataset.

### Important limitation

This is **not** the client’s production data and does **not** represent the exact IV oncology drug verification setting.

That means this repository should be read as a **methodology and pipeline prototype**, not as evidence of operational performance on the client’s real analyser workflow.

## Current pipeline

The pipeline currently supports:

1. loading raw spectral data
2. splitting spectral features and target labels
3. preprocessing spectra by:
   - cropping to the active range
   - SNV normalisation
   - second derivative transformation
4. training and evaluating a classifier
5. saving a classification report
6. exposing inference through a lightweight FastAPI service

The API accepts **raw spectra**, applies preprocessing internally, and returns a predicted compound label.

## Project structure

```text
spectral-drug-verification/
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── results/
│   ├── figures/
│   └── reports/
├── src/
│   ├── features/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   ├── verification/
│   └── pipeline.py
├── tests/
├── run_pipeline.py
└── README.md
```



## How to run

Run the pipeline from the project root:

```bash
python run_pipeline.py
```
run tests:
```
pytest -q
```
Start the API:
```
uvicorn app.api:app --reload
```
Open API docs in the browser:
http://127.0.0.1:8000/docs