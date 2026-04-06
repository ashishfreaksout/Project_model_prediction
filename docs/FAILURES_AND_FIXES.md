# Failures And Fixes

This document records the major problems encountered during the build and how they were resolved.

## 1. Raw Regression Target Histogram Was Misleading

### Problem

For heavily skewed regression targets, the raw histogram compressed almost all values into the far left side of the chart, making it hard to interpret the distribution.

### Fix

- added a default `log1p(target)` distribution view for skewed non-negative targets
- added raw vs log toggle support
- added display-only clipping at the 99th percentile for raw visualization
- added explanatory text so users understand this is a visualization aid, not a data mutation

## 2. Correlation Chart Looked Cramped And Hard To Trust

### Problem

The original feature-correlation chart had label crowding, row clipping, and ambiguous wording.

### Fix

- renamed it to `Top Linear Correlations`
- sorted by absolute correlation strength
- displayed numeric correlation values on bars
- limited visible rows for readability
- made chart height adapt to row count
- added scroll behavior for longer lists
- clarified that these are preliminary linear relationships, not model feature importances

## 3. Results Screen Felt Incomplete

### Problem

The initial post-training dashboard had too little information for a serious ML workflow. Important evaluation views were missing, and one section was empty.

### Fix

- added a results hero with best model, primary metric, RMSE, MAE, and baseline improvement
- added a model leaderboard table
- added actual-vs-predicted, residual diagnostics, error-by-target-range, and feature-influence panels
- added plain-English result summary and interpretation cards

## 4. Winner Selection Looked Wrong

### Problem

The dashboard sometimes appeared to claim one model won while another showed a higher visible `R²`, creating confusion about model selection.

### Root Cause

- model selection used cross-validated metrics
- parts of the UI displayed holdout metrics as if they were the primary selection basis
- regression leaderboard ordering also needed correction

### Fix

- aligned the leaderboard and hero with cross-validated selection logic
- clarified labels such as `CV R²` vs `Holdout R²`
- improved the best-model reason text so the selection basis is explicit

## 5. Manual Prediction Form Asked For Too Many Inputs

### Problem

The manual prediction screen originally showed nearly every uploaded column, including fields that were not useful for the trained model, such as IDs.

### Fix

- generated the form from model-used columns instead of raw dataset columns
- excluded likely IDs, dropped columns, and irrelevant fields
- promoted top inputs first
- added advanced optional inputs section
- added categorical dropdowns for low-cardinality fields
- allowed blank fields so the model pipeline could use defaults and imputers

## 6. Manual Prediction Schema Sometimes Appeared Missing

### Problem

After upgrading the prediction panel, the frontend could show a fallback message saying prediction inputs were not available.

### Root Cause

The running app or currently trained model had been created before the new `prediction_schema` response was available.

### Fix

- confirmed the backend schema generation worked
- restarted the local server
- retrained the model so the updated `/api/train` response included the new schema

## 7. Uploaded Datasets Were Being Saved To Disk

### Problem

The original upload behavior wrote CSVs into an `uploads/` directory, even though the desired product behavior was session-only upload handling.

### Fix

- removed the dataset persistence requirement
- changed upload handling to read CSVs directly from the request stream
- kept data only in the in-memory session store

## 8. Theme Support Was Missing

### Problem

The app had only one visual theme, which made it less flexible for demos and screenshots.

### Fix

- added a full light/dark theme system
- centralized theme tokens with CSS variables
- persisted theme selection in local storage
- made charts and UI components adapt cleanly to theme changes

## 9. Render Deployment Failed

### Problem

The first Render deployment stalled during dependency installation.

### Root Cause

Render defaulted to Python `3.14.3`, which triggered slow source builds for dependencies like NumPy and pandas.

### Fix

- pinned Python to `3.12.3`
- added `.python-version`
- documented and configured the required Render environment settings

## 10. Render Free Tier Was Too Slow For Training

### Problem

Model training on the hosted demo could time out or feel unreliable on the free plan.

### Fix

- reduced training overhead where practical
- documented that free hosting is suitable for a demo, not heavy production training
- identified the next architectural step as background jobs plus persistent storage

## 11. Charts Had Repeated Layout And Clipping Issues

### Problem

Several chart cards had cut-off labels, poor vertical rhythm, or chart areas that did not match the card size.

### Fix

- introduced more content-aware chart sizing
- added better bottom padding and spacing
- allowed bar charts to grow with row count
- improved responsive behavior and card containment

## 12. Project Needed To Stay Dataset-Agnostic

### Problem

As the UI was refined using real datasets, there was a risk of letting dataset-specific language or assumptions leak into the product.

### Fix

- removed domain-specific wording
- kept warnings and summaries driven by metadata, target analysis, and training results
- ensured manual prediction and result summaries come from runtime schema and model outputs rather than fixed dataset assumptions

