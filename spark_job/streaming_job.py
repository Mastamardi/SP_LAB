import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
from joblib import load
import xgboost as xgb
import redis as redis_lib
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
	StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)


# Prometheus metrics
KAFKA_CONSUMED = Counter('kafka_messages_consumed_total', 'Total messages consumed from Kafka')
PROCESSING_LAT_MS = Gauge('spark_processing_latency_ms', 'Spark processing latency per micro-batch in ms')
INFER_LAT_MS = Histogram('model_inference_latency_ms', 'Model inference latency per batch in ms')
PREDICTIONS = Counter('predictions_total', 'Predictions total by label', ['label'])
XGB_PROB_AVG = Gauge('xgb_probability_avg', 'Average XGBoost probability in last batch')
IF_SCORE_G = Gauge('isolation_forest_score', 'Average IsolationForest anomaly score in last batch')
THROUGHPUT = Gauge('throughput_msgs_per_sec', 'Ingest throughput in msgs/sec')
TP = Counter('detection_true_positive_total', 'True positives total')
FP = Counter('detection_false_positive_total', 'False positives total')
FN = Counter('detection_false_negative_total', 'False negatives total')


def get_schema():
	return StructType([
		StructField('transaction_id', StringType(), True),
		StructField('timestamp', StringType(), True),
		StructField('customer_id', StringType(), True),
		StructField('merchant_id', StringType(), True),
		StructField('category', StringType(), True),
		StructField('location', StringType(), True),
		StructField('device_type', StringType(), True),
		StructField('is_international', IntegerType(), True),
		StructField('is_high_risk_country', IntegerType(), True),
		StructField('previous_transactions', IntegerType(), True),
		StructField('avg_transaction_amount', DoubleType(), True),
		StructField('amount', DoubleType(), True),
		StructField('fraudulent', IntegerType(), True),
	])


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--bootstrap', default='localhost:9092')
	p.add_argument('--input_topic', default='transactions')
	p.add_argument('--alerts_topic', default='fraud_alerts')
	p.add_argument('--redis_host', default='localhost')
	p.add_argument('--redis_port', type=int, default=6379)
	p.add_argument('--models_dir', default='models/artifacts')
	p.add_argument('--prometheus_port', type=int, default=8000)
	p.add_argument('--checkpoint', default='spark_job/checkpoints')
	return p.parse_args()


def load_models(models_dir: str):
	if_path = os.path.join(models_dir, 'isolation_forest.joblib')
	xgb_path = os.path.join(models_dir, 'xgboost_model.json')
	enc_path = os.path.join(models_dir, 'encoders.json')

	if_model = load(if_path)
	xgb_model = xgb.XGBClassifier()
	xgb_model.load_model(xgb_path)
	with open(enc_path, 'r') as f:
		enc = json.load(f)
	feature_order = enc['feature_order']
	device_types = enc['device_types']
	categories = enc['categories']
	return if_model, xgb_model, feature_order, device_types, categories


def build_features_pandas(pdf: pd.DataFrame, device_types, categories, feature_order):
	pdf = pdf.copy()
	for col in ['is_international', 'is_high_risk_country', 'previous_transactions', 'fraudulent']:
		pdf[col] = pd.to_numeric(pdf[col], errors='coerce').fillna(0).astype(int)
	for col in ['amount', 'avg_transaction_amount']:
		pdf[col] = pd.to_numeric(pdf[col], errors='coerce').fillna(0.0)

	# Derived
	epsilon = 1e-6
	pdf['amount_to_avg_ratio'] = pdf['amount'] / (pdf['avg_transaction_amount'] + epsilon)

	def parse_ts(x):
		try:
			if isinstance(x, (int, float)):
				return pd.to_datetime(x, unit='ms', utc=True)
			return pd.to_datetime(x, utc=True)
		except Exception:
			return pd.Timestamp.utcnow()

	pdf['ts'] = pdf['timestamp'].apply(parse_ts)
	pdf['hour_of_day'] = pdf['ts'].dt.hour

	for dt in device_types:
		col = f'device_type__{dt}'
		pdf[col] = (pdf['device_type'].fillna('web') == dt).astype(int)
	for cat in categories:
		col = f'category__{cat}'
		pdf[col] = (pdf['category'].fillna('unknown') == cat).astype(int)

	X = pdf.reindex(columns=feature_order, fill_value=0)
	return X.values


def write_to_redis(r: redis_lib.Redis, rows: pd.DataFrame, list_max: int = 50000):
	pipe = r.pipeline(transaction=False)
	for _, row in rows.iterrows():
		key = f"txn:{row['transaction_id']}"
		payload = {k: (None if pd.isna(v) else v) for k, v in row.items()}
		pipe.hset(key, mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in payload.items()})
		pipe.lpush('predictions:latest', json.dumps(payload, default=str))
	pipe.ltrim('predictions:latest', 0, list_max - 1)
	pipe.execute()


def main():
	args = parse_args()
	start_http_server(args.prometheus_port)

	if_model, xgb_model, feature_order, device_types, categories = load_models(args.models_dir)

	spark = (
		SparkSession.builder.appName('FraudStreaming')
		.config('spark.sql.shuffle.partitions', '2')
		.getOrCreate()
	)
	spark.sparkContext.setLogLevel('WARN')

	schema = get_schema()
	df_raw = (
		spark.readStream.format('kafka')
		.option('kafka.bootstrap.servers', args.bootstrap)
		.option('subscribe', args.input_topic)
		.option('startingOffsets', 'latest')
		.option('failOnDataLoss', 'false')
		.load()
	)
	parsed = df_raw.select(F.col('value').cast('string').alias('json_str')) \
		.select(F.from_json('json_str', schema).alias('data')).select('data.*')

	# Basic derived columns in Spark for monitoring
	parsed = parsed.withColumn('event_time', F.to_timestamp('timestamp')) \
		.withColumn('amount_to_avg_ratio', F.col('amount') / (F.col('avg_transaction_amount') + F.lit(1e-6))) \
		.withColumn('hour_of_day', F.hour('event_time'))

	redis_client = redis_lib.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

	last_batch_ts = {'t': time.time(), 'count': 0}

	def process_batch(batch_df, batch_id: int):
		start = time.time()
		rows = batch_df.toPandas()
		KAFKA_CONSUMED.inc(len(rows))
		if rows.empty:
			PROCESSING_LAT_MS.set((time.time() - start) * 1000.0)
			return

		X = build_features_pandas(rows, device_types, categories, feature_order)

		with INFER_LAT_MS.time():
			xgb_probs = xgb_model.predict_proba(X)[:, 1]
			# IsolationForest: lower scores mean anomalies; use decision_function (higher negative => anomaly)
			if_scores = (-if_model.score_samples(X))  # positive anomaly score

		rows['probability_xgb'] = xgb_probs
		rows['prediction_xgb'] = (rows['probability_xgb'] >= 0.5).astype(int)
		rows['if_score'] = if_scores
		anomaly_threshold = float(np.percentile(if_scores, 98))
		rows['is_anomaly_if'] = (rows['if_score'] >= anomaly_threshold).astype(int)
		rows['final_decision'] = ((rows['prediction_xgb'] == 1) | (rows['is_anomaly_if'] == 1)).astype(int)

		# Metrics
		XGB_PROB_AVG.set(float(np.mean(xgb_probs)))
		IF_SCORE_G.set(float(np.mean(if_scores)))

		PREDICTIONS.labels(label='fraud').inc(int(rows['final_decision'].sum()))
		PREDICTIONS.labels(label='legit').inc(int((1 - rows['final_decision']).sum()))

		# If ground truth available
		if 'fraudulent' in rows.columns:
			truth = rows['fraudulent'].astype(int)
			pred = rows['final_decision'].astype(int)
			TP.inc(int(((pred == 1) & (truth == 1)).sum()))
			FP.inc(int(((pred == 1) & (truth == 0)).sum()))
			FN.inc(int(((pred == 0) & (truth == 1)).sum()))

		# Redis sink
		project_cols = [
			'timestamp','transaction_id','customer_id','merchant_id','category','location','device_type',
			'previous_transactions','avg_transaction_amount','amount','prediction_xgb','probability_xgb',
			'if_score','final_decision'
		]
		write_to_redis(redis_client, rows[project_cols])

		# Alerts to Kafka via Spark sink (ensure JSON serializable)
		alerts_rows = rows[rows['final_decision'] == 1]
		if len(alerts_rows) > 0:
			alerts_rows = alerts_rows.copy()
			def to_serializable(rec: Dict[str, Any]) -> Dict[str, Any]:
				out = {}
				for k, v in rec.items():
					if isinstance(v, (pd.Timestamp, datetime)):
						out[k] = v.isoformat()
					else:
						out[k] = v
				return out
			alerts_rows['value'] = alerts_rows.apply(lambda r: json.dumps(to_serializable(r.to_dict()), default=str), axis=1)
			sdf = spark.createDataFrame(alerts_rows[['value']])
			sdf.select(F.col('value').cast('string').alias('value')) \
				.write \
				.format('kafka') \
				.option('kafka.bootstrap.servers', args.bootstrap) \
				.option('topic', args.alerts_topic) \
				.save()

		# Batch metrics
		now = time.time()
		elapsed = now - last_batch_ts['t']
		cnt = len(rows)
		if elapsed > 0:
			THROUGHPUT.set(cnt / elapsed)
		last_batch_ts['t'] = now
		last_batch_ts['count'] = cnt
		PROCESSING_LAT_MS.set((time.time() - start) * 1000.0)

	query = (
		parsed.writeStream
		.outputMode('append')
		.foreachBatch(process_batch)
		.option('checkpointLocation', args.checkpoint)
		.start()
	)

	query.awaitTermination()


if __name__ == '__main__':
	main()
