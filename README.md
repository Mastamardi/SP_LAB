## Realtime Fraud Detection Pipeline (Kafka + Spark + ML + Redis + Prometheus + Grafana + Streamlit)

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Java 8/11 (for Spark)

### Services (Docker)
- Kafka (broker at `localhost:9092`)
- Redis (`localhost:6379`)
- Prometheus (`http://localhost:9090`)
- Grafana (`http://localhost:3000`, admin/admin)

### 1) Start Infra
```bash
docker compose up -d
```

Prometheus is configured to scrape the Spark metrics server from `host.docker.internal:8000` (macOS). If on Linux, change `metrics/prometheus.yml` to point to your host IP.

### 2) Python Env
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Train Models and Save Artifacts
```bash
python models/train_models.py --data Real_fraud_dataset.csv --artifacts_dir models/artifacts
```

### 4) Run Kafka Producer (dataset replay)
```bash
python kafka_producer/producer.py \
  --csv Real_fraud_dataset.csv \
  --bootstrap localhost:9092 \
  --topic transactions \
  --rate 5
```

### 5) Run Spark Streaming Job
Ensure Spark has Kafka package:
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  spark_job/streaming_job.py \
  --bootstrap localhost:9092 \
  --input_topic transactions \
  --alerts_topic fraud_alerts \
  --redis_host localhost \
  --redis_port 6379 \
  --prometheus_port 8000 \
  --models_dir models/artifacts
```

### 6) Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py \
  -- \
  --redis_host localhost \
  --redis_port 6379 \
  --prometheus_url http://localhost:9090
```

### Grafana
1. Open Grafana at `http://localhost:3000` (admin/admin)
2. Add Prometheus data source: URL `http://prometheus:9090`
3. Import dashboard from `grafana/dashboard.json`

### Notes
- Topics are auto-created: `transactions` and `fraud_alerts`.
- Prometheus scrapes every 1s; dashboard refresh set to 1s.
- The Spark job writes predictions to Redis with keys `txn:{transaction_id}` and maintains a list `predictions:latest`.
