# DS_FraudDetection

A collection of Jupyter notebooks, presentation deck and supporting code for exploring, building, and evaluating machine-learning models for fraud detection. The repository contains data exploration, preprocessing, modeling, and evaluation artifacts intended for data scientists and ML practitioners who want a reproducible fraud detection workflow.

## About

This project demonstrates end-to-end steps for a fraud detection data science pipeline: data inspection and cleaning, feature engineering, model training and selection, and evaluation with relevant metrics. The primary artifacts are Jupyter Notebooks (interactive analysis and experiments) and Python scripts (re-usable functions and training pipelines).

## Repository structure

- Presentation deck & Python notebook: include a Jupyter notebook to show machine learning pipeline code, a powerpoint presentation deck to report on the whole process and key findings, a csv file as a raw dataset to experiment with the streamlit app.
- Other files are related to python codes used to develop a Streamlit app to display tho full machine learning pipeline for the fraud detection application.

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

- If the command still fails, use:
python -m streamlit run home.py

- Click on Local URL to open the app

- Notes:
	- Please use the Fraud_segment1.csv in the Presentation deck & Python notebook's folder to upload in the "Upload & Split" page to make sure the model runs correctly.
	- After each step is complete, please click on the current step button to move on to the next step. I will improve the UI in the future to make it a better user experience.

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



