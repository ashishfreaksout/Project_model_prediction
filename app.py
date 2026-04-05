import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from ml_engine import AutomatedML

app = Flask(__name__, static_folder="static")

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File exceeds the maximum allowed size of 200MB."}), 413

# In-memory store for sessions (for demo purposes)
# In production, use Redis and save models to disk/S3
SESSION_STORE = {}

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        df = pd.read_csv(file.stream)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400

    session_id = str(uuid.uuid4())
    # Basic preview
    columns = list(df.columns)
    row_count, col_count = df.shape
    missing_vals = df.isnull().sum().to_dict()
    data_types = df.dtypes.astype(str).to_dict()
    
    # Advanced Insights
    duplicate_rows = int(df.duplicated().sum())
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(exclude=["number"]).columns)
    missing_pct = (df.isnull().mean() * 100).to_dict()
    high_missing = [c for c, p in missing_pct.items() if p > 30]
    
    target_keywords = ["price", "status", "type", "class", "target", "label", "category", "churn", "is_", "has_", "value"]
    suggested_targets = []
    for c in columns:
        cl = c.lower()
        if any(kw in cl for kw in target_keywords):
            t_type = "Regression" if c in num_cols and df[c].nunique(dropna=True) > 20 else "Classification"
            suggested_targets.append({"column": c, "task": t_type})
    
    # Handle NaNs in head data for JSON serialization safely
    head_data = df.head(5).fillna("NaN").to_dict(orient="records")

    SESSION_STORE[session_id] = {
        "dataframe": df,
        "ml_engine": AutomatedML(df)
    }

    return jsonify({
        "session_id": session_id,
        "filename": file.filename,
        "rows": row_count,
        "columns": columns,
        "missing": missing_vals,
        "types": data_types,
        "head": head_data,
        "duplicates": duplicate_rows,
        "numeric_cols": len(num_cols),
        "categorical_cols": len(cat_cols),
        "high_missing_cols": high_missing,
        "suggested_targets": suggested_targets[:6]
    })

@app.route("/api/target_insight", methods=["POST"])
def target_insight():
    data = request.json
    session_id = data.get("session_id")
    target_column = data.get("target_column")
    
    if not session_id or session_id not in SESSION_STORE:
        return jsonify({"error": "Invalid session"}), 400
        
    engine = SESSION_STORE[session_id]["ml_engine"]
    try:
        engine.set_target(target_column)
        return jsonify({
            "task_type": engine.task_type,
            "target_summary": engine.target_summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/analyze", methods=["POST"])
def analyze_data():
    data = request.json
    session_id = data.get("session_id")
    target_column = data.get("target_column")

    if not session_id or session_id not in SESSION_STORE:
        return jsonify({"error": "Invalid session"}), 400

    engine = SESSION_STORE[session_id]["ml_engine"]
    
    try:
        engine.set_target(target_column)
        eda_results = engine.run_eda()
        SESSION_STORE[session_id]["task_type"] = engine.task_type
        return jsonify({
            "task_type": engine.task_type,
            "target": target_column,
            "eda": eda_results,
            "target_summary": engine.target_summary,
            "preprocessing_plan": engine.get_preprocessing_plan(),
            "candidate_models": engine.get_candidate_model_list(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/train", methods=["POST"])
def train_models():
    data = request.json
    session_id = data.get("session_id")
    use_log_transform = data.get("use_log_transform", False)

    if not session_id or session_id not in SESSION_STORE:
        return jsonify({"error": "Invalid session"}), 400

    engine = SESSION_STORE[session_id]["ml_engine"]
    
    try:
        engine.set_training_options(use_log_target=use_log_transform)
        engine.train_all_models()
        results = engine.get_model_comparison()
        best_model_info = engine.get_best_model_info()
        return jsonify({
            "models": results,
            "best_model": best_model_info,
            "feature_importance": engine.get_feature_importance(),
            "evaluation_charts": engine.get_evaluation_charts(),
            "prediction_schema": engine.get_prediction_schema(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    session_id = data.get("session_id")
    features = data.get("features", {})

    if not session_id or session_id not in SESSION_STORE:
        return jsonify({"error": "Invalid session"}), 400

    engine = SESSION_STORE[session_id]["ml_engine"]
    try:
        # Features is a dict of column -> value
        prediction = engine.predict(features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5001")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
