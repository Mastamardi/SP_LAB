import argparse
import json
import time
import csv
from datetime import datetime
from kafka.producer.kafka import KafkaProducer


def parse_args():
	parser = argparse.ArgumentParser(description="Replay CSV into Kafka as a stream")
	parser.add_argument('--csv', required=True, help='Path to CSV file')
	parser.add_argument('--bootstrap', default='localhost:9092', help='Kafka bootstrap servers')
	parser.add_argument('--topic', default='transactions', help='Kafka topic name')
	parser.add_argument('--rate', type=float, default=5.0, help='Messages per second')
	parser.add_argument('--loop', action='store_true', help='Loop the dataset endlessly')
	return parser.parse_args()


def create_producer(bootstrap_servers: str) -> KafkaProducer:
	return KafkaProducer(
		bootstrap_servers=bootstrap_servers,
		value_serializer=lambda v: json.dumps(v).encode('utf-8'),
		key_serializer=lambda v: str(v).encode('utf-8') if v is not None else None,
	)


def row_to_message(row: dict) -> dict:
	# Ensure schema and types
	def to_int(x):
		try:
			return int(x)
		except Exception:
			return 0
	def to_float(x):
		try:
			return float(x)
		except Exception:
			return 0.0

	# timestamp normalization
	ts = row.get('timestamp')
	if ts and isinstance(ts, str) and ts.isdigit():
		# epoch ms
		try:
			iso_ts = datetime.utcfromtimestamp(int(ts) / 1000.0).strftime('%Y-%m-%dT%H:%M:%SZ')
		except Exception:
			iso_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
	else:
		iso_ts = ts or datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

	msg = {
		"transaction_id": str(row.get('transaction_id')),
		"timestamp": iso_ts,
		"customer_id": str(row.get('customer_id')),
		"merchant_id": str(row.get('merchant_id')),
		"category": row.get('category') or "unknown",
		"location": row.get('location') or "unknown",
		"device_type": row.get('device_type') or "web",
		"is_international": to_int(row.get('is_international')),
		"is_high_risk_country": to_int(row.get('is_high_risk_country')),
		"previous_transactions": to_int(row.get('previous_transactions')),
		"avg_transaction_amount": to_float(row.get('avg_transaction_amount')),
		"amount": to_float(row.get('amount')),
		"fraudulent": to_int(row.get('fraudulent')) if row.get('fraudulent') is not None else 0,
	}
	return msg


def main():
	args = parse_args()
	producer = create_producer(args.bootstrap)
	interval = 1.0 / max(args.rate, 0.1)

	def send_rows():
		with open(args.csv, 'r', newline='') as f:
			reader = csv.DictReader(f)
			for row in reader:
				msg = row_to_message(row)
				key = msg.get('transaction_id')
				producer.send(args.topic, value=msg, key=key)
				producer.flush()
				time.sleep(interval)

	if args.loop:
		while True:
			send_rows()
	else:
		send_rows()

	producer.close()


if __name__ == '__main__':
	main()
