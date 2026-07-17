# DS_FraudDetection

A collection of Jupyter notebooks, presentation deck and supporting code for exploring, building, and evaluating machine-learning models for fraud detection. The repository contains data exploration, preprocessing, modeling, and evaluation artifacts intended for data scientists and ML practitioners who want a reproducible fraud detection workflow.

## About

This project demonstrates end-to-end steps for a fraud detection data science pipeline: data inspection and cleaning, feature engineering, model training and selection, and evaluation with relevant metrics. The primary artifacts are Jupyter Notebooks (interactive analysis and experiments) and Python scripts (re-usable functions and training pipelines).

## Repository structure

Typical layout (not all folders may exist depending on this repository's current state):

- notebooks/ or root .ipynb files — Jupyter notebooks for exploration and experiments
- data/ — (recommended) place for raw and processed datasets (NOT included in repo unless committed)
- src/ — reusable Python modules and helper functions
- models/ — saved model artifacts and serialized pipelines
- outputs/ or reports/ — figures, model evaluation reports, and logs

Adjust these paths to match the actual files in the repository.

## Getting started

Prerequisites

- Python 3.8+ recommended
- JupyterLab or Jupyter Notebook
- Virtual environment (venv, conda) recommended

Install dependencies

If a requirements.txt exists in the repo:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

If there's no requirements file, install common data science packages used for fraud detection:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab xgboost lightgbm
```

Data

- Place your dataset files in the `data/` directory (create it if missing). Use descriptive names such as `transactions.csv`, `labels.csv`, or `fraud_data.csv`.
- For privacy and compliance, do not commit sensitive or personally identifiable information (PII) to this repository.

## Notebooks and recommended workflow

Work through the notebooks in this order for a reproducible analysis:

1. Data exploration and visualization — inspect distributions, missing values, and class imbalance
2. Preprocessing and feature engineering — cleaning, encoding, scaling, and deriving new features
3. Modeling — train a selection of models (logistic regression, tree-based models, boosting)
4. Evaluation and interpretation — compute metrics (precision, recall, F1, ROC AUC, PR AUC), confusion matrices, and calibration

Each notebook contains narrative cells explaining the steps and code cells to run experiments. Use JupyterLab to run cells sequentially and save outputs.

## Model training & evaluation

- Use stratified sampling or cross-validation adapted for imbalanced datasets.
- Prefer metrics that capture performance on the minority (fraud) class: precision-recall AUC, recall at a fixed precision, F1-score.
- Consider resampling strategies (SMOTE, undersampling), class weighting, or threshold tuning if class imbalance is severe.

## Reproducing results

- Set a random seed where applicable to make experiments reproducible (notebooks usually include a `RANDOM_STATE` variable).
- Record package versions (pip freeze > requirements.txt) or use a lockfile for environment reproducibility.
- If running long training jobs, save trained models to `models/` and log experiment metadata (hyperparameters, metrics) to `outputs/` or a small CSV.

## Contributing

- Open an issue for bug reports or feature requests.
- If you want to contribute code, create a feature branch, add tests if appropriate, and open a pull request describing your changes.
- Keep notebooks tidy: clear large outputs before committing and avoid committing large datasets or model binaries.

## Notes on data privacy

Fraud datasets often contain sensitive information. Ensure you have permission to use and share any datasets. When publishing results, anonymize or aggregate data to avoid exposing PII.

## License

If you have a preferred license, add a LICENSE file. If unsure, consider the MIT License for permissive reuse.

## Contact

Maintainer: HongNgocDuong (GitHub)

If you'd like the README customized further (for example, to include specific notebook names, example commands to run particular notebooks, or a requirements file generated from the environment), tell me which details to include and I will update the README accordingly.

## Running the Streamlit app

Quick steps to run the Streamlit web app included in this repo.

- Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .env\Scripts\activate    # Windows (PowerShell)
```

- Install dependencies (if `requirements.txt` exists) and Streamlit:

```bash
pip install --upgrade pip
pip install -r requirements.txt  # optional if file exists
pip install streamlit
```

- Run the app (use the repository root as working directory):

```bash
streamlit run home.py
```

- Common options:

```bash
streamlit run home.py --server.port 8501        # run on specific port
streamlit run home.py --server.headless true    # run without opening a browser
```

- Notes:
	- If this repo uses a multi-page layout, Streamlit will detect pages in the `pages/` directory automatically.
	- To share the app publicly, consider deploying to Streamlit Cloud or another hosting provider.

