# NexusML

NexusML is a full-stack AutoML web application for tabular CSV datasets. A user uploads a dataset, chooses a target column, reviews automated pre-training analysis, benchmarks multiple machine learning models, compares results in a polished dashboard, and then runs manual predictions against the trained best model.

It is designed as a portfolio-grade project that demonstrates:
- full-stack product thinking
- practical machine learning workflow design
- frontend dashboarding and UX refinement
- deployment and debugging across local and hosted environments

## What This Project Does

In simple terms, NexusML helps someone answer:

"If I upload a spreadsheet, what can I predict from it, which model works best, and how can I test predictions without writing code?"

The app:
1. accepts a CSV upload
2. inspects the dataset and suggests likely target columns
3. detects whether the problem is classification or regression
4. analyzes data quality, missingness, skew, duplicates, and basic relationships
5. trains a benchmark set of ML models
6. selects the strongest model using validation metrics
7. explains the result with charts, metrics, and feature influence
8. lets the user simulate predictions through a generated input form

## Non-Technical Explanation

If you were showing this to a non-technical stakeholder, you could describe it like this:

> NexusML is a smart analysis tool for spreadsheet data. You upload a CSV, tell it what outcome you care about, and it automatically checks the data, tries several prediction models, shows which one performed best, and gives you a simple interface to test new predictions.

## Why I Built It

This project was built to turn a typical machine learning notebook workflow into a usable product experience. Instead of stopping at model code, it carries the process through:
- upload
- target selection
- automated analysis
- training
- evaluation
- interpretation
- manual prediction
- deployment

## Core Features

- CSV upload with preview and target suggestions
- Automatic task detection for classification vs regression
- Pre-training analysis with target summary, data quality checks, preprocessing plan, and model candidates
- Regression-specific improvements such as skew warnings and optional `log1p(target)` transform
- Multi-model benchmarking with cross-validation and holdout metrics
- Post-training results dashboard with leaderboard, comparison chart, diagnostics, and feature influence
- Manual prediction panel generated from the trained model's actual input schema
- Theme toggle with persistent light and dark mode
- Render-ready deployment configuration

## Example User Flow

1. Upload a CSV dataset.
2. Select a target column.
3. Review the analysis screen to understand data quality and likely preprocessing.
4. Train candidate models.
5. Inspect the results dashboard to see which model won and why.
6. Open manual prediction mode and test new examples.

## Tech Stack

- Backend: Flask
- Data and ML: pandas, NumPy, SciPy, scikit-learn, XGBoost
- Frontend: HTML, CSS, vanilla JavaScript, Chart.js
- Deployment target: Render

## Local Setup

### Requirements

- Python 3.12

### Install

```bash
pip install -r requirements.txt
```

### Run locally

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

## Deployment

This repo includes:
- [render.yaml](./render.yaml)
- [requirements.txt](./requirements.txt)
- [.python-version](./.python-version)

The app is intended to be deployed as a Render `Web Service`, not a static site.

See [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) for exact settings and caveats.

## Interview / Portfolio Positioning

This project is strong when presented as:
- a productized AutoML workflow for tabular business datasets
- a demonstration of both ML engineering and frontend UX
- an example of turning a model pipeline into an end-to-end user-facing application

For interview-ready talking points, see [docs/INTERVIEW_GUIDE.md](./docs/INTERVIEW_GUIDE.md).

## Screenshots

Selected product screens are documented in [docs/SCREENSHOTS.md](./docs/SCREENSHOTS.md).

### Results Dashboard

![Results dashboard hero](./docs/screenshots/results-hero.png)

### Analysis And Diagnostics

![Analysis overview](./docs/screenshots/analysis-overview.png)
![Target distribution and top correlations](./docs/screenshots/target-distribution-and-correlations.png)

## Project Structure

```text
.
├── app.py
├── ml_engine.py
├── requirements.txt
├── render.yaml
├── static/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
└── docs/
    ├── API.md
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    ├── FAILURES_AND_FIXES.md
    ├── INTERVIEW_GUIDE.md
    ├── SCREENSHOTS.md
    └── screenshots/
```

## Important Limitations

- Session state is stored in memory for demo simplicity.
- Uploaded datasets are not persisted across server restarts.
- Training runs synchronously in the web process.
- Free hosting is fine for a demo, but not ideal for heavy user-driven training workloads.

These tradeoffs are documented more fully in:
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)
- [docs/FAILURES_AND_FIXES.md](./docs/FAILURES_AND_FIXES.md)
