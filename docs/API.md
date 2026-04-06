# API Reference

## Base URL

Local development:

```text
http://127.0.0.1:5001
```

## `POST /api/upload`

Uploads a CSV and initializes a new session.

### Request

- `multipart/form-data`
- field: `file`

### Returns

- `session_id`
- `filename`
- row and column counts
- column list
- missing values
- inferred dtypes
- preview rows
- duplicate count
- suggested target columns

### Notes

- only CSV files are accepted
- file size limit is 200 MB
- uploaded data is stored only in the current in-memory session

## `POST /api/target_insight`

Gets a quick summary for the selected target.

### Request JSON

```json
{
  "session_id": "uuid",
  "target_column": "your_target"
}
```

### Returns

- `task_type`
- `target_summary`

## `POST /api/analyze`

Runs automated pre-training analysis.

### Request JSON

```json
{
  "session_id": "uuid",
  "target_column": "your_target"
}
```

### Returns

- `task_type`
- `target`
- `eda`
- `target_summary`
- `preprocessing_plan`
- `candidate_models`

## `POST /api/train`

Trains all candidate models and returns the evaluation package.

### Request JSON

```json
{
  "session_id": "uuid",
  "use_log_transform": true
}
```

### Returns

- `models`
- `best_model`
- `feature_importance`
- `evaluation_charts`
- `prediction_schema`

### Notes

- `use_log_transform` is only meaningful for supported regression targets
- metrics are returned on the original target scale even when log transform is used internally

## `POST /api/predict`

Runs a single manual prediction using the trained best model.

### Request JSON

```json
{
  "session_id": "uuid",
  "features": {
    "feature_a": 12,
    "feature_b": "category_value"
  }
}
```

### Returns

```json
{
  "prediction": 123.45
}
```

or for classification:

```json
{
  "prediction": "Survived"
}
```

### Notes

- partial input is supported
- missing fields are aligned and handled by the preprocessing pipeline

