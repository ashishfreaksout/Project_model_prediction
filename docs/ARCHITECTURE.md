# Architecture Overview

## High-Level System

NexusML is a single-service web application:

- `app.py` exposes the Flask routes and stores session state in memory.
- `ml_engine.py` contains the ML workflow: task inference, EDA, preprocessing, model training, evaluation, and prediction.
- `static/index.html`, `static/css/styles.css`, and `static/js/app.js` implement the single-page frontend.

## Request Flow

### 1. Upload

`POST /api/upload`

- reads the uploaded CSV into pandas directly from the request stream
- builds a lightweight preview and metadata summary
- creates a session ID
- stores the dataframe and `AutomatedML` engine instance in `SESSION_STORE`

### 2. Target Selection

`POST /api/target_insight`

- sets the target column
- infers task type
- returns a target summary

### 3. Pre-Training Analysis

`POST /api/analyze`

- runs EDA on the selected target
- returns:
  - task type
  - target summary
  - warnings and suggestions
  - preprocessing plan
  - candidate model list
  - chart-ready analysis payloads

### 4. Training

`POST /api/train`

- applies optional training settings such as `log1p(target)` for supported regression targets
- preprocesses the data
- benchmarks candidate models using cross-validation
- evaluates each model on a holdout split
- selects the best model
- returns model comparison data, best-model explanation, evaluation charts, feature importance, and manual-prediction schema

### 5. Manual Prediction

`POST /api/predict`

- accepts a partial feature payload
- aligns the provided fields to the trained model input schema
- lets the preprocessing pipeline handle defaults and imputations
- returns a single predicted value or class

## ML Pipeline Design

## Task Detection

The engine infers:
- `classification` when the target is boolean, non-numeric, or low-cardinality numeric
- `regression` when the numeric target has enough continuous variation

## Column Role Detection

The engine identifies:
- numeric columns
- categorical columns
- datetime-like columns
- long-text columns
- ID-like columns

This allows the app to stay dataset-agnostic while still making practical preprocessing decisions.

## Preprocessing

Numeric features:
- median imputation
- standard scaling

Categorical features:
- most-frequent imputation
- one-hot encoding

Excluded by default:
- likely IDs
- datetime columns
- long free-text columns

## Candidate Models

Classification:
- Baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN
- Naive Bayes
- XGBoost when available

Regression:
- Baseline
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN
- XGBoost when available

## Model Selection Logic

The best model is selected from cross-validated performance, not from whichever holdout metric happens to look best on one split.

For regression, the ranking prioritizes:
1. `cv_r2_mean`
2. `cv_rmse_mean`
3. `holdout_r2`

This choice makes the model selection more stable and less likely to overreact to a single test split.

## Manual Prediction Schema

The prediction form is generated from the trained model's actual used input columns, not from every uploaded dataset column.

The schema includes:
- key inputs
- optional inputs
- categorical dropdown options for low-cardinality fields
- default values based on training data
- excluded columns list

This keeps the UI cleaner and closer to what the model truly uses.

## Frontend Design

The frontend is a single-page interface with these major screens:
- upload and dataset preview
- pre-training analysis
- post-training results dashboard
- manual prediction panel

Notable frontend refinements:
- theme-aware light/dark mode
- dynamic chart sizing
- content-aware chart cards
- model-specific result summaries
- dataset-agnostic chart labels and warnings

## Current Architecture Tradeoffs

Good for:
- demos
- portfolio presentation
- small to medium interactive experiments

Not yet designed for:
- multi-user persistence
- long-running background jobs
- saved models across restarts
- production-scale concurrent training

Those upgrades are possible, but would require a queue, persistent storage, and a more durable session architecture.

