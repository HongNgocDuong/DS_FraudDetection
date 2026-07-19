# DS_FraudDetection

This repository contains a compact fraud-detection demo project with a Streamlit app, supporting Python scripts, and a presentation notebook. The app now runs in a static demo mode that uses a bundled sample dataset, so it is easy to review without uploading files or retraining models.

## What is in this repository

- Streamlit app entry point: [home.py](home.py)
- Static demo data loader: [static_demo.py](static_demo.py)
- Workflow navigation helpers: [workflow_nav.py](workflow_nav.py)
- Demo pages: [pages/01_upload_split.py](pages/01_upload_split.py), [pages/02_preprocess_cv.py](pages/02_preprocess_cv.py), [pages/03_model_tuning.py](pages/03_model_tuning.py), and [pages/04_evaluate.py](pages/04_evaluate.py)
- Notebook and presentation materials: [Presentation deck & Python notebook](Presentation%20deck%20&%20Python%20notebook)

## Project goals

The project demonstrates a fraud-detection workflow covering:

1. Dataset inspection
2. Preprocessing and cross-validation setup
3. Model tuning overview
4. Final evaluation metrics and summary outputs

The current app is designed as a polished static showcase rather than a fully interactive training pipeline.

## Getting started

### Prerequisites

- Python 3.9+ recommended
- A virtual environment is recommended

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the app

Via URL: https://mlfrauddetection.streamlit.app/

From the repository root:

```bash
.venv\Scripts\python.exe -m streamlit run home.py
```

The app will open at http://localhost:8501/ and display the preloaded demo results.

## Demo behavior

The current Streamlit experience is intentionally static:

- The sample dataset is already loaded
- Each page shows precomputed results
- No upload, training, or evaluation step is required to view the demo

If the bundled CSV file is missing or empty, the app uses a built-in fallback dataset so the demo continues to run.

## Notebook and presentation materials

The notebook and presentation deck in [Presentation deck & Python notebook](Presentation%20deck%20&%20Python%20notebook) can be used for additional context, screenshots, and narrative explanation of the workflow.

## Notes

- Keep the repository tidy by avoiding large raw datasets or model binaries in version control.
- For privacy and compliance, do not commit sensitive or personally identifiable information.



