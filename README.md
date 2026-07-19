# DS_FraudDetection

This repository contains a fraud-detection demo built with Streamlit, a presentation deck, and a notebook that walks through the modeling workflow. The app is now set up as a polished, static demo that uses a bundled sample dataset so it can be viewed without uploading files.

## What is in this repository

- Presentation deck & Python notebook: notebook walkthrough, presentation slides, and the bundled sample fraud dataset
- Streamlit app: a read-only demo experience with multiple workflow pages
- Supporting Python code: data-loading and demo-state logic for the app

## Demo app overview

The Streamlit app is designed for presentation and walkthrough purposes. It does not require the user to upload data or run training steps interactively.

The current flow includes:

1. Dataset overview
2. Preprocessing and cross-validation summary
3. Model tuning summary
4. Evaluation metrics and results

The app uses the bundled file:

- Presentation deck & Python notebook/fraud_sample_dataset.csv

## Project structure

- home.py: landing page for the demo
- workflow_nav.py: sidebar navigation for the workflow pages
- static_demo.py: shared logic that loads the bundled dataset and initializes the demo state
- pages/: workflow pages for the demo experience
- Presentation deck & Python notebook/: notebook, slides, and sample CSV

## Getting started

### Prerequisites

- Python 3.9+
- A virtual environment is recommended

### Setup

Run the following from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the Streamlit app
- Via URL: https://mlfrauddetection.streamlit.app/

- Via Github:

```powershell
.\.venv\Scripts\python.exe -m streamlit run home.py
```

Then open the local URL shown in the terminal.

## Notes on the demo data

- The demo uses the bundled CSV as the default preloaded dataset.
- No upload is required to view the sample workflow.
- The page content is intentionally static for demonstration purposes.

## Notes on data privacy

Fraud datasets can contain sensitive information. Only use data that you are authorized to inspect or share.

