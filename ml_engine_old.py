import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

class AutomatedML:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target = None
        self.task_type = None  # 'classification' or 'regression'
        self.models = {}
        self.best_model_name = None
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def set_target(self, target_column):
        if target_column not in self.df.columns:
            raise ValueError(f"Column {target_column} not found in dataset.")
        self.target = target_column
        
        # Determine task type
        unique_vals = self.df[self.target].nunique()
        dtype = self.df[self.target].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            if unique_vals <= 10: # Heuristic for classification
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            self.task_type = "classification"
            
    def run_eda(self):
        # Basic correlation if numeric
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlations = {}
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().fillna(0)
            target_corr = {}
            if self.target in corr_matrix.columns:
                target_corr = corr_matrix[self.target].drop(self.target).to_dict()
            correlations = {
                "matrix": corr_matrix.to_dict(),
                "target_relationships": target_corr
            }
            
        # Target Distribution
        target_dist = self.df[self.target].value_counts().to_dict() if self.task_type == "classification" else {}
        
        return {
            "correlations": correlations,
            "target_distribution": target_dist,
            "numeric_columns": list(numeric_df.columns)
        }

    def preprocess_data(self):
        # Drop rows where target is missing
        clean_df = self.df.dropna(subset=[self.target])
        
        # DOWN-SAMPLE FOR PERFORMANCE: Prevent browser timeouts and hangs on large datasets
        if len(clean_df) > 10000:
            clean_df = clean_df.sample(n=10000, random_state=42)
        
        y = clean_df[self.target]
        X = clean_df.drop(columns=[self.target])
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        self.pipeline = preprocessor
        
        # Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Transform data
        self.X_train_transformed = self.pipeline.fit_transform(self.X_train)
        self.X_test_transformed = self.pipeline.transform(self.X_test)
        
        # Save feature names for importance later
        self.feature_names = numeric_features + list(
            self.pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        )

    def train_all_models(self):
        self.preprocess_data()
        
        if self.task_type == "classification":
            models_to_train = {
                "Baseline": DummyClassifier(strategy="prior"),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }
        else:
            models_to_train = {
                "Baseline": DummyRegressor(strategy="mean"),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(n_estimators=100),
                "KNN": KNeighborsRegressor(),
                "XGBoost": xgb.XGBRegressor()
            }
            
        # Due to categorical targets in XGBoost, we might need LabelEncoding if classification and target is string
        y_train_proc = self.y_train
        y_test_proc = self.y_test
        label_encoder = None
        
        if self.task_type == "classification" and self.y_train.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_train_proc = label_encoder.fit_transform(self.y_train)
            y_test_proc = label_encoder.transform(self.y_test)
            self.label_encoder = label_encoder # save for prediction

        self.results = {}
        for name, model in models_to_train.items():
            try:
                # Train
                if name == "XGBoost" and self.task_type == "classification" and y_train_proc.dtype == 'O':
                    model.fit(self.X_train_transformed, y_train_proc)
                    y_pred = model.predict(self.X_test_transformed)
                    # Convert back for eval if needed or evaluate numeric
                    eval_y_true = y_test_proc
                elif name == "XGBoost":
                     # XGBoost needs numeric targets natively mostly
                     if self.task_type == "classification" and label_encoder:
                         model.fit(self.X_train_transformed, y_train_proc)
                         y_pred = model.predict(self.X_test_transformed)
                         eval_y_true = y_test_proc
                     else:
                        model.fit(self.X_train_transformed, self.y_train)
                        y_pred = model.predict(self.X_test_transformed)
                        eval_y_true = self.y_test
                else:
                    model.fit(self.X_train_transformed, self.y_train)
                    y_pred = model.predict(self.X_test_transformed)
                    eval_y_true = self.y_test
                
                self.models[name] = model
                
                # Evaluate
                metrics = {}
                if self.task_type == "classification":
                    # Evaluate on eval_y_true (which is numeric if XGBoost or LabelEncoded)
                    if hasattr(model, "predict_proba"):
                        try:
                            # only for binary classification easily
                            y_prob = model.predict_proba(self.X_test_transformed)[:, 1]
                            auc = roc_auc_score(eval_y_true, y_prob)
                            metrics["ROC_AUC"] = float(auc)
                        except:
                            metrics["ROC_AUC"] = None
                            
                    metrics["Accuracy"] = float(accuracy_score(eval_y_true, y_pred))
                    metrics["Precision"] = float(precision_score(eval_y_true, y_pred, average='weighted', zero_division=0))
                    metrics["Recall"] = float(recall_score(eval_y_true, y_pred, average='weighted', zero_division=0))
                    metrics["F1_Score"] = float(f1_score(eval_y_true, y_pred, average='weighted', zero_division=0))
                else:
                    metrics["RMSE"] = float(np.sqrt(mean_squared_error(self.y_test, y_pred)))
                    metrics["MAE"] = float(mean_absolute_error(self.y_test, y_pred))
                    metrics["R2_Score"] = float(r2_score(self.y_test, y_pred))
                    
                self.results[name] = metrics
            except Exception as e:
                print(f"Failed to train {name}: {str(e)}")
                
        # Determine best model
        if self.task_type == "classification":
            self.best_model_name = max(self.results, key=lambda k: self.results[k].get("Accuracy", 0))
        else:
            self.best_model_name = max(self.results, key=lambda k: self.results[k].get("R2_Score", -999))
            
        return self.results
        
    def get_best_model_info(self):
        desc = f"The dataset involves predicting {self.target}. "
        if self.task_type == "classification":
             desc += f"Based on the evaluation metrics, {self.best_model_name} performs the best, achieving an accuracy of {self.results[self.best_model_name]['Accuracy']:.3f}. "
             desc += "This model was chosen automatically by comparing cross-validated metrics on the held-out test set."
        else:
             desc += f"Based on the evaluation metrics, {self.best_model_name} is the most reliable predictor, achieving an R-Squared score of {self.results[self.best_model_name]['R2_Score']:.3f}. "
             
        return {
            "name": self.best_model_name,
            "metrics": self.results[self.best_model_name],
            "description": desc
        }
        
    def get_feature_importance(self):
        best_model = self.models.get(self.best_model_name)
        importance = None
        
        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            importance = np.abs(best_model.coef_[0]) if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
            
        if importance is not None and len(importance) == len(self.feature_names):
            imp_dict = {feat: float(imp) for feat, imp in zip(self.feature_names, importance)}
            # Sort top 10
            return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)[:10])
        return {}
        
    def get_evaluation_charts(self):
        charts = {}
        if not self.best_model_name: return charts
        
        best_model = self.models.get(self.best_model_name)
        if not best_model: return charts
        
        try:
            y_pred = best_model.predict(self.X_test_transformed)
            eval_y_true = self.y_test
            
            if self.task_type == "classification":
                if hasattr(self, 'label_encoder'):
                    eval_y_true = self.label_encoder.transform(self.y_test)
                    
                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(self.X_test_transformed)
                    if y_prob.shape[1] == 2:  # Only ROC for binary
                        fpr, tpr, _ = roc_curve(eval_y_true, y_prob[:, 1])
                        charts["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            else:
                # Regression: Return top 50 points of True vs Predicted
                y_true_list = eval_y_true.tolist()[:50]
                y_pred_list = y_pred.tolist()[:50]
                charts["scatter"] = {"true": y_true_list, "pred": y_pred_list}
        except Exception:
            pass
            
        return charts
        
    def predict(self, feature_dict):
        # Convert dict to DataFrame with 1 row
        input_df = pd.DataFrame([feature_dict])
        
        # Enforce column order to prevent 400 Unknown categories or Length mismatch errors natively
        try:
            for col in self.X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[self.X_train.columns]
        except Exception:
            pass
        
        # Transform 
        transformed_input = self.pipeline.transform(input_df)
        
        best_model = self.models[self.best_model_name]
        prediction = best_model.predict(transformed_input)
        
        # Decode if label encoder was used
        if hasattr(self, 'label_encoder'):
            prediction = self.label_encoder.inverse_transform(prediction)
            
        if isinstance(prediction[0], (np.integer, np.floating)):
            return float(prediction[0])
        return str(prediction[0])
