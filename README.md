# Phishing Behavioral Cybersecurity Research Project

This project provides a complete framework for the research paper "The Role of Human Factors in Phishing Attacks: A Behavioral Cybersecurity Study." It includes the data, analysis scripts, machine learning pipeline, and visualization tools necessary to reproduce the findings.

## Project Structure

- `final.tex`: The final LaTeX source for the research paper.
- `config/`: Configuration files for the project.
- `data/`: Contains raw, processed, and synthetic data.
- `src/`: All Python source code.
  - `processing/`: Scripts for data cleaning and feature engineering.
  - `analysis/`: The statistical analysis script.
  - `models/`: The machine learning training pipeline.
  - `visualization/`: The interactive Streamlit dashboard.
- `tests/`: Unit and integration tests for the codebase.
- `requirements.txt`: A list of all necessary Python packages.
- `results/`: Output directory for generated reports and figures.
- `outputs/`: Output directory for paper figures.

## How to Run

### 1. Setup Environment
Ensure you have Python 3.9+ installed, then create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
To regenerate all results from scratch, run the following scripts in order:

**a. Preprocess the data:**
```bash
python src/processing/preprocess_for_modeling.py
```

**b. Run the statistical analysis:**
```bash
python src/analysis/statistical_analysis.py
```

**c. Train the machine learning model:**
```bash
python src/models/train_vulnerability_predictor.py
```

### 3. View the Interactive Dashboard
To explore the results interactively, launch the Streamlit dashboard:
```bash
streamlit run src/visualization/dashboard.py
```