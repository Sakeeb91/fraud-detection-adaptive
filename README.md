# Fraud Detection System with Adaptive Thresholding

A real-time fraud scoring system for payment transactions that adapts to evolving fraud patterns, handles severe class imbalance (1:10000), and optimizes for business-specific cost matrices with concept drift detection.

## Overview

This system combines unsupervised anomaly detection with supervised gradient boosting to identify fraudulent transactions in real-time. It features adaptive thresholding that responds to changing fraud patterns and a champion-challenger deployment model for continuous improvement.

## Key Features

- **Extreme Class Imbalance Handling**: Designed for 1:10000 fraud ratios using specialized sampling and custom loss functions
- **Hybrid Detection**: Combines Isolation Forest and One-Class SVM for unsupervised scoring with XGBoost for supervised classification
- **Cost-Sensitive Learning**: Custom loss function encoding business-specific fraud costs (false negative vs false positive tradeoffs)
- **Adaptive Thresholding**: Dynamic threshold optimization using precision-recall curves with cost weighting
- **Concept Drift Detection**: Statistical tests monitoring feature distribution shifts over time
- **Model Versioning**: Champion-challenger deployment with automatic rollback capabilities

## Architecture

```
                    +------------------+
                    |  Transaction     |
                    |  Input Stream    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Feature Engine   |
                    | - Velocity       |
                    | - Graph-based    |
                    | - Behavioral     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +------------------+          +------------------+
     | Unsupervised     |          | Supervised       |
     | Anomaly Scoring  |          | Classification   |
     | - Isolation Forest          | - XGBoost        |
     | - One-Class SVM  |          | - Custom Loss    |
     +--------+---------+          +--------+---------+
              |                             |
              +-------------+---------------+
                            |
                            v
                   +------------------+
                   | Score Fusion &   |
                   | Adaptive         |
                   | Thresholding     |
                   +--------+---------+
                            |
                            v
                   +------------------+
                   | Drift Detection  |
                   | & Model Updates  |
                   +------------------+
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| ML Framework | scikit-learn, XGBoost | Core modeling |
| Drift Detection | alibi-detect | Distribution monitoring |
| Graph Features | NetworkX | Transaction network analysis |
| Model Versioning | MLflow | Experiment tracking, champion-challenger |
| API Layer | FastAPI | Real-time scoring endpoint |
| Data Validation | Pydantic | Input/output schemas |

## Project Structure

```
fraud-detection-adaptive/
├── src/
│   ├── features/           # Feature engineering modules
│   ├── models/             # Model definitions and training
│   ├── scoring/            # Real-time scoring logic
│   ├── drift/              # Drift detection components
│   └── api/                # FastAPI application
├── tests/                  # Test suite
├── configs/                # Configuration files
├── docs/                   # Documentation
└── notebooks/              # Exploration and analysis
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda for package management

### Installation

```bash
git clone https://github.com/Sakeeb91/fraud-detection-adaptive.git
cd fraud-detection-adaptive
pip install -r requirements.txt
```

### Quick Start

```python
from src.scoring import FraudScorer

scorer = FraudScorer.load("models/champion")
result = scorer.predict(transaction_data)
print(f"Fraud Score: {result.score}, Decision: {result.decision}")
```

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Detailed phased development plan
- [Feature Engineering Guide](docs/FEATURES.md) - Feature definitions and rationale
- [Model Architecture](docs/MODELS.md) - Model design decisions

## License

MIT License

## Author

Sakeeb Rahman (rahman.sakeeb@gmail.com)
