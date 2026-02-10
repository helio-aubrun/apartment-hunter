# Apartment Hunter - Price Prediction

Machine learning application for predicting apartment prices with a Flask API, web interface, and Docker support.

## Features

- Three ML models: Linear Regression, Random Forest, XGBoost
- REST API and web interface
- Docker deployment ready
- Model persistence
- Dataset management

## Quick Start

### Docker (Recommended)

```bash
docker-compose up -d
```

Open http://localhost:5000

### Local Python

```bash
conda activate sklearn_env
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Usage

### Python Script

```python
from predict_price import PricePredictor
import pandas as pd

df = pd.read_csv("clean_dataset.csv")
predictor = PricePredictor(df)
predictor.train("RandomForest")
print(predictor.get_scores())
```

### REST API

```bash
# Load data
curl -X POST http://localhost:5000/api/load-data \
  -H "Content-Type: application/json" \
  -d '{"filepath": "clean_dataset.csv", "target": "buy_price"}'

# Train model
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"models": ["RandomForest"]}'

# Predict
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "RandomForest", "data": {"surface": 100, "rooms": 3}}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/status` | GET | Application status |
| `/api/load-data` | POST | Load dataset |
| `/api/train` | POST | Train models |
| `/api/predict` | POST | Make predictions |
| `/api/scores` | GET | Model performance |

See [README_FLASK.md](README_FLASK.md) for complete API documentation.

## Model Performance

| Model | RMSE |
|-------|------|
| Random Forest | ~21,475 |
| XGBoost | ~36,042 |
| Linear Regression | ~287,160 |

## Project Structure

```
apartment-hunter/
├── app.py                 # Flask application
├── predict_price.py       # ML predictor
├── clean_dataset.csv      # Dataset
├── requirements.txt       # Dependencies
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker config
└── templates/
    └── index.html        # Web UI
```

## Documentation

- **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** - Docker quick start
- **[README_DOCKER.md](README_DOCKER.md)** - Docker details
- **[README_FLASK.md](README_FLASK.md)** - API documentation

## Requirements

- Python 3.12+
- pandas, numpy, scikit-learn, xgboost, flask

Or just use Docker - no Python installation needed.

## Docker Commands

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## Configuration

Edit hyperparameters in `predict_price.py`:

```python
self.params = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    }
}
```

## Troubleshooting

**Port 5000 in use?**
```bash
# Change port in docker-compose.yml or app.py
```

**Docker not starting?**
```bash
docker-compose logs
```

**Module not found?**
```bash
pip install -r requirements.txt
```

## License

[Your License]

## Author

[Your Name]

---

For detailed documentation, see the README files in the [Documentation](#documentation) section.