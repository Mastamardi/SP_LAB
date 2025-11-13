import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from joblib import dump


FEATURE_BASE = [
	'amount',
	'avg_transaction_amount',
	'previous_transactions',
	'is_international',
	'is_high_risk_country',
]


def build_features(df: pd.DataFrame, categories_meta: Dict[str, List[str]] | None = None):
	# Clean and cast
	df = df.copy()
	for col in ['is_international', 'is_high_risk_country', 'previous_transactions', 'fraudulent']:
		df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
	for col in ['amount', 'avg_transaction_amount']:
		df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

	# Derived features
	epsilon = 1e-6
	df['amount_to_avg_ratio'] = df['amount'] / (df['avg_transaction_amount'] + epsilon)

	# Timestamp derived
	def parse_ts(x):
		try:
			if isinstance(x, (int, float)):
				return pd.to_datetime(x, unit='ms', utc=True)
			return pd.to_datetime(x, utc=True)
		except Exception:
			return pd.Timestamp.utcnow()

	df['ts'] = df['timestamp'].apply(parse_ts)
	df['hour_of_day'] = df['ts'].dt.hour
	df['day_of_week'] = df['ts'].dt.dayofweek

	# Categorical encodings
	device_types = categories_meta.get('device_types') if categories_meta else sorted(df['device_type'].fillna('web').unique().tolist())
	categories = categories_meta.get('categories') if categories_meta else sorted(df['category'].fillna('unknown').unique().tolist())

	for dt in device_types:
		col = f'device_type__{dt}'
		df[col] = (df['device_type'].fillna('web') == dt).astype(int)
	for cat in categories:
		col = f'category__{cat}'
		df[col] = (df['category'].fillna('unknown') == cat).astype(int)

	feature_cols = FEATURE_BASE + [
		'amount_to_avg_ratio',
		'hour_of_day',
		# day_of_week could be added if desired
	] + [f'device_type__{d}' for d in device_types] + [f'category__{c}' for c in categories]

	return df, feature_cols, {'device_types': device_types, 'categories': categories}


def train_iforest(X: np.ndarray, random_state: int = 42) -> IsolationForest:
	model = IsolationForest(
		n_estimators=200,
		contamination='auto',
		random_state=random_state,
		n_jobs=-1,
	)
	model.fit(X)
	return model


def train_xgb(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> xgb.XGBClassifier:
	clf = xgb.XGBClassifier(
		n_estimators=300,
		max_depth=5,
		learning_rate=0.1,
		subsample=0.8,
		colsample_bytree=0.8,
		eval_metric='logloss',
		n_jobs=-1,
		random_state=random_state,
	)
	clf.fit(X, y)
	return clf


def evaluate(y_true, y_pred, y_prob):
	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, zero_division=0)
	rec = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred, zero_division=0)
	try:
		auc = roc_auc_score(y_true, y_prob)
	except Exception:
		auc = float('nan')
	cm = confusion_matrix(y_true, y_pred).tolist()
	return {
		"accuracy": acc,
		"precision": prec,
		"recall": rec,
		"f1": f1,
		"roc_auc": auc,
		"confusion_matrix": cm,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', required=True)
	parser.add_argument('--artifacts_dir', default='models/artifacts')
	args = parser.parse_args()

	os.makedirs(args.artifacts_dir, exist_ok=True)

	df = pd.read_csv(args.data)
	# Build with discovered categories
	df_feat, feature_cols, cats = build_features(df)

	X = df_feat[feature_cols].values
	y = df_feat['fraudulent'].fillna(0).astype(int).values

	# Train/test split for XGB evaluation
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	# Train models
	if_model = train_iforest(X_train)
	xgb_model = train_xgb(X_train, y_train)

	# Evaluate
	y_prob = xgb_model.predict_proba(X_test)[:, 1]
	y_pred = (y_prob >= 0.5).astype(int)
	metrics = evaluate(y_test, y_pred, y_prob)
	print("Evaluation:", json.dumps(metrics, indent=2))

	# Save artifacts
	dump(if_model, os.path.join(args.artifacts_dir, 'isolation_forest.joblib'))
	xgb_model.save_model(os.path.join(args.artifacts_dir, 'xgboost_model.json'))
	with open(os.path.join(args.artifacts_dir, 'encoders.json'), 'w') as f:
		json.dump({
			"device_types": cats['device_types'],
			"categories": cats['categories'],
			"feature_order": feature_cols,
		}, f, indent=2)
	with open(os.path.join(args.artifacts_dir, 'metrics.json'), 'w') as f:
		json.dump(metrics, f, indent=2)

	print(f"Artifacts saved to {args.artifacts_dir}")


if __name__ == '__main__':
	main()
