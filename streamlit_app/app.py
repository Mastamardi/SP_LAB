import argparse
import json
import time
from typing import List, Dict, Any

import pandas as pd
import redis as redis_lib
import requests
import streamlit as st


def parse_args():
	# Streamlit passes args after '--'
	import sys
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--redis_host', default='localhost')
	parser.add_argument('--redis_port', type=int, default=6379)
	parser.add_argument('--prometheus_url', default='http://localhost:9090')
	args, _ = parser.parse_known_args(sys.argv)
	return args


def fetch_latest(r: redis_lib.Redis, limit: int = 10000) -> pd.DataFrame:
	items = r.lrange('predictions:latest', 0, limit - 1)
	rows: List[Dict[str, Any]] = []
	for it in reversed(items):  # oldest first
		try:
			rows.append(json.loads(it))
		except Exception:
			continue
	return pd.DataFrame(rows)


def prom_instant(prom_url: str, metric: str) -> float:
	try:
		r = requests.get(f"{prom_url}/api/v1/query", params={'query': metric}, timeout=2)
		data = r.json()
		res = data.get('data', {}).get('result', [])
		if not res:
			return 0.0
		return float(res[0]['value'][1])
	except Exception:
		return 0.0


def main():
	args = parse_args()
	r = redis_lib.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

	st.set_page_config(page_title='Fraud Detection Live', layout='wide')
	st.title('Realtime Fraud Detection')

	# Controls
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		rate = st.slider('Replay rate (msg/s)', min_value=1, max_value=50, value=5, step=1)
		r.set('producer:rate', rate)
	with col2:
		mode = st.selectbox('Inference Mode', options=['ensemble', 'xgb_only', 'if_only'], index=0)
		r.set('inference:mode', mode)
	with col3:
		display_mode = st.selectbox('Display Mode', options=['sequential_play', 'rolling_window'], index=0)
	with col4:
		window_size = st.slider('Window size', min_value=50, max_value=1000, value=300, step=50)

	# State for sequential play history
	if 'seq_index' not in st.session_state:
		st.session_state.seq_index = None
	if 'history' not in st.session_state:
		st.session_state.history = []  # newest first
	if 'last_txn_id' not in st.session_state:
		st.session_state.last_txn_id = None

	# Layout placeholders
	current_ph = st.empty()
	history_ph = st.empty()
	metrics_ph = st.empty()

	def style_df(df: pd.DataFrame) -> Any:
		def highlight(row):
			color = 'background-color: #ffcccc' if int(row.get('final_decision', 0)) == 1 else ''
			return [color] * len(row)
		return df.style.apply(highlight, axis=1)

	if display_mode == 'sequential_play':
		llen = r.llen('predictions:latest')
		if llen == 0:
			current_ph.info('Waiting for data...')
		else:
			if st.session_state.seq_index is None:
				st.session_state.seq_index = llen - 1  # start at oldest
			else:
				if st.session_state.seq_index > 0:
					st.session_state.seq_index -= 1
				else:
					st.session_state.seq_index = 0

			item = r.lindex('predictions:latest', st.session_state.seq_index)
			row = {}
			try:
				row = json.loads(item) if item else {}
			except Exception:
				row = {}

			if row:
				# Update history if new
				txn_id = row.get('transaction_id')
				if txn_id and txn_id != st.session_state.last_txn_id:
					st.session_state.history.insert(0, row)
					st.session_state.history = st.session_state.history[:window_size]
					st.session_state.last_txn_id = txn_id

				current_df = pd.DataFrame([row])
				current_ph.subheader('Current (processing)')
				current_ph.dataframe(style_df(current_df), width='stretch', height=130)

				hist_df = pd.DataFrame(st.session_state.history[1:]) if len(st.session_state.history) > 1 else pd.DataFrame()
				if not hist_df.empty:
					history_ph.subheader('Processed (history)')
					history_ph.dataframe(style_df(hist_df), width='stretch', height=400)
				else:
					history_ph.info('No history yet')
			else:
				current_ph.info('Waiting for data...')
	else:
		# Rolling window: split current and history
		df = fetch_latest(r, limit=10000)
		if not df.empty:
			if 'timestamp' in df.columns:
				df = df.sort_values('timestamp', ascending=False)
			current_df = df.head(1)
			hist_df = df.iloc[1:window_size]
			current_ph.subheader('Current (latest)')
			current_ph.dataframe(style_df(current_df), width='stretch', height=130)
			if not hist_df.empty:
				history_ph.subheader('Recent (history)')
				history_ph.dataframe(style_df(hist_df), width='stretch', height=400)
		else:
			current_ph.info('Waiting for data...')

	# Metrics
	throughput = prom_instant(args.prometheus_url, 'throughput_msgs_per_sec')
	latency = prom_instant(args.prometheus_url, 'spark_processing_latency_ms')
	infer = prom_instant(args.prometheus_url, 'rate(model_inference_latency_ms_count[5s])')
	acc_tp = prom_instant(args.prometheus_url, 'detection_true_positive_total')
	acc_fp = prom_instant(args.prometheus_url, 'detection_false_positive_total')
	acc_fn = prom_instant(args.prometheus_url, 'detection_false_negative_total')

	prec = (acc_tp / (acc_tp + acc_fp)) if (acc_tp + acc_fp) > 0 else 0.0
	rec = (acc_tp / (acc_tp + acc_fn)) if (acc_tp + acc_fn) > 0 else 0.0
	with metrics_ph.container():
		m1, m2, m3, m4, m5 = st.columns(5)
		m1.metric('Throughput', f"{throughput:.2f} msgs/s")
		m2.metric('Processing Latency', f"{latency:.1f} ms")
		m3.metric('Inference Rate', f"{infer:.2f} /s")
		m4.metric('Precision', f"{prec:.2f}")
		m5.metric('Recall', f"{rec:.2f}")

	time.sleep(1)
	st.rerun()


if __name__ == '__main__':
	main()
