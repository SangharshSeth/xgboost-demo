# XGBoost Fraud Detection Demo

Fraud detection for Indian bank transactions using XGBoost.

## Features
- **500 customers** with realistic profiles (salaried, self-employed, students, business, retired)
- **200K+ transactions** over 12 months with proper balance tracking
- **0.1% fraud rate** with realistic patterns (account takeover, card cloning, unusual merchants)
- **35+ features** via PySpark (temporal, velocity, behavioral, balance features)
- **CPU/GPU training** with XGBoost
- **REST API** for real-time scoring

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2. Generate transaction data (~2 min)
uv run src/data_generator.py

# 3. Feature engineering (requires pyspark - see note below)
uv run src/feature_engineering.py

# 4. Train model
uv run src/train.py

# 5. Start API
uv run src/api.py
```

## API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TXN001","customer_id":"CUST001","amount":15000,"transaction_type":"UPI","merchant_name":"Amazon","merchant_category":"ecommerce","city":"Mumbai","timestamp":"2024-02-04T10:30:00","balance_before":50000,"device_id":"DEV001"}'
```

## Project Structure

```
src/
├── data_generator.py       # Generate 200K realistic transactions
├── feature_engineering.py  # PySpark feature creation (35+ features)
├── train.py                # XGBoost training with metrics
└── api.py                  # FastAPI scoring service
data/                       # Generated parquet files
models/                     # Trained model & artifacts
```

## Note on PySpark (Windows)

PySpark installation can be slow on Windows. If `uv add pyspark` hangs, try:
```powershell
# Install Java first (required for Spark)
winget install Oracle.JDK.21

# Then install pyspark
uv add pyspark
```

## GPU Training (Optional)

```powershell
uv run src/train.py --gpu
```
