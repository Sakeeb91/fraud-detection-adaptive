# Implementation Plan: Fraud Detection System with Adaptive Thresholding

## Expert Role

**Role**: ML Engineer (Anomaly Detection & Financial ML Specialist)

**Rationale**: This project requires deep expertise in:
- Extreme class imbalance handling (1:10000 ratio)
- Unsupervised anomaly detection ensembles
- Custom loss function design for gradient boosting
- Concept drift detection and monitoring
- Production ML patterns (champion-challenger deployment)

This is fundamentally an applied ML engineering challenge in the financial domain.

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRAUD DETECTION SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │ Transaction  │    │              FEATURE ENGINE                       │  │
│  │   Input      │───▶│  ┌────────────┬────────────┬─────────────────┐   │  │
│  │   Stream     │    │  │ Velocity   │ Graph      │ Behavioral      │   │  │
│  └──────────────┘    │  │ Features   │ Features   │ Anomaly         │   │  │
│                      │  │            │ (NetworkX) │ Features        │   │  │
│                      │  └────────────┴────────────┴─────────────────┘   │  │
│                      └──────────────────────┬───────────────────────────┘  │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        SCORING ENGINE                                 │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐      │  │
│  │  │   UNSUPERVISED LAYER    │    │    SUPERVISED LAYER         │      │  │
│  │  │  ┌─────────────────┐    │    │  ┌─────────────────────┐    │      │  │
│  │  │  │ Isolation Forest│    │    │  │ XGBoost with        │    │      │  │
│  │  │  │ (contamination= │    │    │  │ Custom Loss         │    │      │  │
│  │  │  │  auto)          │    │    │  │ (cost-sensitive)    │    │      │  │
│  │  │  └────────┬────────┘    │    │  └──────────┬──────────┘    │      │  │
│  │  │  ┌────────▼────────┐    │    │             │               │      │  │
│  │  │  │ One-Class SVM   │    │    │             │               │      │  │
│  │  │  │ (nu=0.001)      │    │    │             │               │      │  │
│  │  │  └────────┬────────┘    │    │             │               │      │  │
│  │  └───────────┼─────────────┘    └─────────────┼───────────────┘      │  │
│  │              │                                │                       │  │
│  │              └────────────┬───────────────────┘                       │  │
│  │                           ▼                                           │  │
│  │              ┌─────────────────────────┐                              │  │
│  │              │ Score Fusion Layer      │                              │  │
│  │              │ (weighted ensemble)     │                              │  │
│  │              └────────────┬────────────┘                              │  │
│  └───────────────────────────┼──────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    DECISION ENGINE                                    │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐      │  │
│  │  │ Adaptive Threshold      │    │ Business Rules              │      │  │
│  │  │ Optimizer               │◀──▶│ Engine                      │      │  │
│  │  │ (cost-weighted P-R)     │    │ (overrides, whitelists)     │      │  │
│  │  └─────────────────────────┘    └─────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MONITORING & ADAPTATION                            │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────┐      │  │
│  │  │ Drift Detector          │    │ Champion-Challenger         │      │  │
│  │  │ (alibi-detect)          │    │ Model Manager               │      │  │
│  │  │ - KS Test               │    │ (MLflow)                    │      │  │
│  │  │ - MMD                   │    │                             │      │  │
│  │  └─────────────────────────┘    └─────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Transaction ──▶ Validation ──▶ Feature Extraction ──▶ Model Inference ──▶ Decision
     │              │                  │                    │               │
     │              ▼                  ▼                    ▼               ▼
     │         [Schema Check]    [Feature Store]     [Score Log]     [Action Log]
     │                                 │                    │               │
     │                                 └────────────────────┴───────────────┘
     │                                                      │
     │                                                      ▼
     └─────────────────────────────────────────────▶ [Drift Monitor]
                                                            │
                                                            ▼
                                                    [Retrain Trigger]
```

---

## Technology Choices

### Core Stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| **ML Framework** | scikit-learn | >=1.3.0 | Industry standard, excellent documentation, familiar API |
| **Gradient Boosting** | XGBoost | >=2.0.0 | Native custom objective support, proven in Kaggle fraud competitions |
| **Drift Detection** | alibi-detect | >=0.11.0 | Purpose-built, supports KS/MMD tests, free |
| **Graph Analysis** | NetworkX | >=3.1 | Lightweight, beginner-friendly, sufficient for feature extraction |
| **Model Versioning** | MLflow | >=2.8.0 | Free local mode, handles model registry, champion-challenger |
| **API Layer** | FastAPI | >=0.104.0 | Async-native, automatic OpenAPI docs, Pydantic integration |
| **Data Validation** | Pydantic + Pandera | >=2.5.0 | Runtime type checking, DataFrame validation |

### Tradeoffs Considered

| Decision | Alternative | Why Current Choice |
|----------|-------------|-------------------|
| XGBoost over LightGBM | LightGBM | XGBoost has simpler custom loss API, better documented |
| NetworkX over PyG | PyTorch Geometric | Overkill for feature extraction; NetworkX is simpler |
| alibi-detect over custom | Hand-rolled tests | Proven implementation, saves development time |
| FastAPI over Flask | Flask | Native async, auto-docs, modern Python patterns |

### Fallback Options

| Primary | Fallback | Trigger Condition |
|---------|----------|-------------------|
| XGBoost custom loss | sklearn class_weight | If custom loss proves too complex |
| alibi-detect MMD | Simple KS test | If MMD is too slow for real-time |
| MLflow registry | JSON-based versioning | If MLflow setup is problematic |

---

## Phased Implementation Plan

### Phase 0: Project Foundation
**Goal**: Establish development infrastructure and synthetic data generation

**Scope**:
- `src/__init__.py` - Package initialization
- `src/data/synthetic.py` - Synthetic transaction generator
- `src/data/schemas.py` - Pydantic models for transactions
- `configs/default.yaml` - Configuration schema
- `tests/conftest.py` - Pytest fixtures

**Deliverables**:
- [ ] Synthetic dataset generator producing 1:10000 class ratio
- [ ] Transaction schema with validation
- [ ] Configuration loading utility
- [ ] Test fixtures for all subsequent phases

**Verification**:
```bash
pytest tests/test_data_generation.py -v
python -c "from src.data.synthetic import generate_transactions; print(generate_transactions(1000).shape)"
```

**Technical Challenges**:
- Generating realistic synthetic fraud patterns
- Ensuring reproducibility with random seeds

**Definition of Done**:
- `generate_transactions(n)` returns DataFrame with expected columns
- Fraud ratio is within 5% of target (1:10000)
- All tests pass

---

### Phase 1: Feature Engineering Pipeline
**Goal**: Implement all feature extraction modules

**Scope**:
- `src/features/velocity.py` - Transaction velocity features
- `src/features/graph.py` - Network-based features
- `src/features/behavioral.py` - Behavioral anomaly features
- `src/features/pipeline.py` - Unified feature pipeline

**Deliverables**:
- [ ] Velocity features: txn count, amount sum in 1h/24h/7d windows
- [ ] Graph features: degree centrality, PageRank, clustering coefficient
- [ ] Behavioral features: deviation from user baseline
- [ ] Combined pipeline with caching

**Verification**:
```bash
pytest tests/test_features.py -v
python -c "from src.features.pipeline import FeaturePipeline; p = FeaturePipeline(); print(p.get_feature_names())"
```

**Technical Challenges**:
- Efficient window calculations without full history
- Graph construction from transaction pairs
- Handling cold-start users with no history

**Definition of Done**:
- Pipeline transforms raw transactions to feature matrix
- All feature values are finite (no NaN/inf)
- Feature extraction completes in <100ms per transaction

**Code Skeleton**:
```python
# src/features/velocity.py
from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class VelocityConfig:
    windows: list[int] = (3600, 86400, 604800)  # 1h, 24h, 7d in seconds

class VelocityFeatureExtractor:
    def __init__(self, config: VelocityConfig = None):
        self.config = config or VelocityConfig()

    def extract(self, transaction: dict, history: pd.DataFrame) -> Dict[str, float]:
        """Extract velocity features for a single transaction."""
        raise NotImplementedError
```

---

### Phase 2: Unsupervised Anomaly Detection
**Goal**: Implement Isolation Forest and One-Class SVM ensemble

**Scope**:
- `src/models/isolation_forest.py` - IF wrapper with auto-contamination
- `src/models/one_class_svm.py` - OCSVM wrapper
- `src/models/unsupervised_ensemble.py` - Score fusion logic

**Deliverables**:
- [ ] Isolation Forest with contamination='auto'
- [ ] One-Class SVM with nu=0.0001 (matching fraud rate)
- [ ] Weighted score fusion (configurable weights)
- [ ] Anomaly score normalization to [0, 1]

**Verification**:
```bash
pytest tests/test_unsupervised.py -v
```

**Technical Challenges**:
- OCSVM scaling issues with large datasets (use SGD variant if needed)
- Score calibration between different detectors
- Memory management for IF on large feature sets

**Definition of Done**:
- Ensemble returns scores in [0, 1] range
- Known anomalies (synthetic frauds) score higher than median
- Inference time <50ms per transaction

**Code Skeleton**:
```python
# src/models/unsupervised_ensemble.py
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class UnsupervisedEnsemble:
    def __init__(self, if_weight: float = 0.6, svm_weight: float = 0.4):
        self.if_model = IsolationForest(contamination='auto', random_state=42)
        self.svm_model = OneClassSVM(nu=0.0001, kernel='rbf')
        self.if_weight = if_weight
        self.svm_weight = svm_weight
        self.scaler = MinMaxScaler()

    def fit(self, X: np.ndarray) -> 'UnsupervisedEnsemble':
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1], higher = more anomalous."""
        raise NotImplementedError
```

---

### Phase 3: Supervised Model with Custom Loss
**Goal**: Implement XGBoost with cost-sensitive custom objective

**Scope**:
- `src/models/cost_matrix.py` - Cost matrix definition
- `src/models/custom_loss.py` - XGBoost custom objective
- `src/models/xgboost_classifier.py` - Training wrapper

**Deliverables**:
- [ ] Cost matrix: FN costs 100x more than FP
- [ ] Custom gradient/hessian for XGBoost
- [ ] Training with SMOTE or class weighting
- [ ] Cross-validation with stratified folds

**Verification**:
```bash
pytest tests/test_supervised.py -v
```

**Technical Challenges**:
- Deriving correct gradient/hessian for weighted loss
- Preventing XGBoost from ignoring minority class
- Hyperparameter tuning with custom objective

**Debugging Scenarios**:
- If model predicts all zeros: increase FN cost, check class weights
- If gradient is NaN: check for log(0) in loss function
- If no convergence: reduce learning rate, check for data issues

**Definition of Done**:
- Model achieves >0.5 recall on fraud class
- Precision-recall AUC > baseline (random)
- Custom loss correctly penalizes FN >> FP

**Code Skeleton**:
```python
# src/models/custom_loss.py
import numpy as np

def weighted_binary_loss(y_true: np.ndarray, y_pred: np.ndarray,
                         fn_cost: float = 100.0, fp_cost: float = 1.0):
    """
    Custom loss where false negatives cost more than false positives.

    For XGBoost, we need to return gradient and hessian.
    """
    # Clip predictions to avoid log(0)
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Weighted cross-entropy
    weights = np.where(y_true == 1, fn_cost, fp_cost)

    # Gradient: d(loss)/d(pred)
    grad = weights * (y_pred - y_true)

    # Hessian: d2(loss)/d(pred)2
    hess = weights * y_pred * (1 - y_pred)

    return grad, hess
```

---

### Phase 4: Threshold Optimization
**Goal**: Implement adaptive threshold selection based on cost-weighted metrics

**Scope**:
- `src/scoring/threshold_optimizer.py` - P-R curve analysis
- `src/scoring/cost_calculator.py` - Business cost computation
- `src/scoring/scorer.py` - Unified scoring interface

**Deliverables**:
- [ ] Cost-weighted F-score calculation
- [ ] Optimal threshold finder
- [ ] Threshold update mechanism
- [ ] Score-to-decision logic

**Verification**:
```bash
pytest tests/test_threshold.py -v
```

**Technical Challenges**:
- Choosing between point estimate vs. range
- Handling threshold instability with small validation sets
- Real-time threshold updates vs. batch recalculation

**Definition of Done**:
- Threshold minimizes expected cost on validation set
- Threshold selection is reproducible
- API returns both score and binary decision

**Code Skeleton**:
```python
# src/scoring/threshold_optimizer.py
from sklearn.metrics import precision_recall_curve
import numpy as np

class ThresholdOptimizer:
    def __init__(self, fn_cost: float = 100.0, fp_cost: float = 1.0):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.optimal_threshold = 0.5

    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Find threshold that minimizes expected cost."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Calculate cost at each threshold
        # Cost = FN_cost * (1 - recall) * P(fraud) + FP_cost * (1 - precision) * P(legit)
        raise NotImplementedError
```

---

### Phase 5: Drift Detection
**Goal**: Implement feature distribution monitoring and drift alerts

**Scope**:
- `src/drift/detector.py` - Statistical test wrappers
- `src/drift/monitor.py` - Continuous monitoring logic
- `src/drift/alerts.py` - Alert generation

**Deliverables**:
- [ ] KS test for univariate drift
- [ ] MMD test for multivariate drift
- [ ] Reference distribution storage
- [ ] Drift severity scoring

**Verification**:
```bash
pytest tests/test_drift.py -v
```

**Technical Challenges**:
- Setting appropriate p-value thresholds
- Handling multiple testing correction
- MMD computational cost for high dimensions

**Definition of Done**:
- Drift detected within 1000 samples of distribution shift
- False positive rate <5% on stable distributions
- Drift report includes affected features

**Code Skeleton**:
```python
# src/drift/detector.py
from alibi_detect.cd import KSDrift, MMDDrift
import numpy as np

class DriftDetector:
    def __init__(self, p_val_threshold: float = 0.05):
        self.p_val_threshold = p_val_threshold
        self.ks_detector = None
        self.mmd_detector = None
        self.reference_data = None

    def fit(self, reference_data: np.ndarray) -> 'DriftDetector':
        """Set reference distribution for drift detection."""
        self.reference_data = reference_data
        self.ks_detector = KSDrift(reference_data, p_val=self.p_val_threshold)
        return self

    def detect(self, current_data: np.ndarray) -> dict:
        """Check if current data has drifted from reference."""
        raise NotImplementedError
```

---

### Phase 6: Model Versioning & Deployment
**Goal**: Implement champion-challenger model management and API

**Scope**:
- `src/models/registry.py` - MLflow model registry wrapper
- `src/models/challenger.py` - A/B testing logic
- `src/api/main.py` - FastAPI application
- `src/api/routes.py` - Scoring endpoints

**Deliverables**:
- [ ] Model registration with MLflow
- [ ] Champion-challenger comparison
- [ ] Automatic promotion logic
- [ ] REST API for scoring

**Verification**:
```bash
pytest tests/test_api.py -v
uvicorn src.api.main:app --reload
curl -X POST localhost:8000/score -d '{"amount": 100, ...}'
```

**Technical Challenges**:
- Stateless API with model caching
- Handling model loading latency
- Thread-safety for concurrent requests

**Definition of Done**:
- API returns scores in <200ms p99
- Champion model auto-loads on startup
- Challenger can be promoted via API call

**Code Skeleton**:
```python
# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.scoring.scorer import FraudScorer

app = FastAPI(title="Fraud Detection API")
scorer = None

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant_id: str
    user_id: str
    timestamp: str
    # ... other fields

class ScoringResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    decision: str  # "approve", "review", "decline"
    model_version: str

@app.on_event("startup")
async def load_model():
    global scorer
    scorer = FraudScorer.load_champion()

@app.post("/score", response_model=ScoringResponse)
async def score_transaction(request: TransactionRequest):
    raise NotImplementedError
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning Signs | Mitigation |
|------|------------|--------|---------------------|------------|
| OCSVM too slow for large data | High | Medium | Training >1hr on 100k samples | Switch to SGDOneClassSVM or remove from ensemble |
| Custom XGBoost loss incorrect | Medium | High | Model outputs all 0 or all 1 | Add unit tests with known inputs/outputs; fall back to class_weight |
| Drift detector false positives | Medium | Medium | Alerts on stable data | Tune p-value threshold; add Bonferroni correction |
| Class imbalance defeats model | Medium | High | Recall < 0.1 on fraud class | Increase SMOTE ratio; try different sampling strategies |
| MLflow setup complexity | Low | Low | Installation errors | Fall back to JSON-based model versioning |
| Feature engineering too slow | Medium | Medium | >500ms per transaction | Pre-compute features; use approximate algorithms |

### Contingency Cuts

If running over time, prioritize in this order (cut from bottom):
1. **Must Have**: Phases 0-3 (foundation, features, unsupervised, supervised)
2. **Should Have**: Phase 4 (threshold optimization)
3. **Nice to Have**: Phase 5 (drift detection)
4. **Stretch**: Phase 6 (full API, champion-challenger)

---

## Testing Strategy

### Test Pyramid

```
                    /\
                   /  \
                  / E2E \      <- 2-3 tests (API flow)
                 /________\
                /          \
               / Integration \  <- 5-10 tests (pipeline tests)
              /______________\
             /                \
            /    Unit Tests    \  <- 50+ tests (all functions)
           /____________________\
```

### Testing Framework

- **Framework**: pytest
- **Coverage Target**: 80%
- **CI**: GitHub Actions

### First Three Tests

1. **test_synthetic_data_ratio** - Verify fraud ratio is 1:10000
```python
def test_synthetic_data_ratio():
    from src.data.synthetic import generate_transactions
    df = generate_transactions(100000, fraud_ratio=0.0001)
    actual_ratio = df['is_fraud'].mean()
    assert 0.00008 < actual_ratio < 0.00012  # Within 20% of target
```

2. **test_velocity_features_shape** - Verify feature output dimensions
```python
def test_velocity_features_shape():
    from src.features.velocity import VelocityFeatureExtractor
    extractor = VelocityFeatureExtractor()
    features = extractor.extract(sample_transaction, sample_history)
    assert len(features) == 6  # count + amount for 3 windows
```

3. **test_unsupervised_scores_bounded** - Verify score normalization
```python
def test_unsupervised_scores_bounded():
    from src.models.unsupervised_ensemble import UnsupervisedEnsemble
    ensemble = UnsupervisedEnsemble()
    ensemble.fit(training_data)
    scores = ensemble.score(test_data)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
```

---

## First Concrete Task

### File to Create
`src/data/synthetic.py`

### Function Signature
```python
def generate_transactions(
    n_transactions: int,
    fraud_ratio: float = 0.0001,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic fraud patterns.

    Args:
        n_transactions: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (default 1:10000)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns:
        - transaction_id: str
        - timestamp: datetime
        - user_id: str
        - merchant_id: str
        - amount: float
        - is_fraud: int (0 or 1)
    """
```

### Starter Code

```python
# src/data/synthetic.py
"""Synthetic transaction data generator for fraud detection model development."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_transactions(
    n_transactions: int,
    fraud_ratio: float = 0.0001,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic fraud patterns.

    Fraud transactions have different characteristics:
    - Higher amounts (mean $500 vs $50 for legitimate)
    - Unusual hours (more likely 2-5 AM)
    - New merchants (not in user's history)

    Args:
        n_transactions: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (default 1:10000)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with transaction data including fraud labels
    """
    np.random.seed(random_state)

    # Calculate fraud count
    n_fraud = max(1, int(n_transactions * fraud_ratio))
    n_legit = n_transactions - n_fraud

    # Generate user and merchant pools
    n_users = max(100, n_transactions // 100)
    n_merchants = max(50, n_transactions // 200)

    users = [f"user_{i:06d}" for i in range(n_users)]
    merchants = [f"merchant_{i:04d}" for i in range(n_merchants)]

    # Generate legitimate transactions
    legit_data = {
        'transaction_id': [f"txn_{i:012d}" for i in range(n_legit)],
        'timestamp': _generate_timestamps(n_legit, random_state),
        'user_id': np.random.choice(users, n_legit),
        'merchant_id': np.random.choice(merchants, n_legit),
        'amount': np.random.exponential(scale=50, size=n_legit),  # Mean $50
        'is_fraud': np.zeros(n_legit, dtype=int)
    }

    # Generate fraudulent transactions
    fraud_data = {
        'transaction_id': [f"txn_{i:012d}" for i in range(n_legit, n_transactions)],
        'timestamp': _generate_timestamps(n_fraud, random_state + 1, fraud=True),
        'user_id': np.random.choice(users, n_fraud),
        'merchant_id': np.random.choice(merchants, n_fraud),
        'amount': np.random.exponential(scale=500, size=n_fraud),  # Mean $500
        'is_fraud': np.ones(n_fraud, dtype=int)
    }

    # Combine and shuffle
    legit_df = pd.DataFrame(legit_data)
    fraud_df = pd.DataFrame(fraud_data)
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Round amounts to cents
    df['amount'] = df['amount'].round(2)

    return df


def _generate_timestamps(
    n: int,
    random_state: int,
    fraud: bool = False
) -> pd.DatetimeIndex:
    """Generate transaction timestamps."""
    np.random.seed(random_state)

    base_date = datetime(2024, 1, 1)

    if fraud:
        # Fraud more likely at unusual hours (2-5 AM)
        hours = np.random.choice([2, 3, 4, 5], n)
    else:
        # Normal distribution centered on business hours
        hours = np.clip(np.random.normal(14, 4, n), 0, 23).astype(int)

    days = np.random.randint(0, 365, n)
    minutes = np.random.randint(0, 60, n)

    timestamps = [
        base_date + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(days, hours, minutes)
    ]

    return pd.to_datetime(timestamps)


if __name__ == "__main__":
    # Quick test
    df = generate_transactions(10000)
    print(f"Generated {len(df)} transactions")
    print(f"Fraud ratio: {df['is_fraud'].mean():.6f}")
    print(df.head())
```

### Verification Method
```bash
cd fraud-detection-adaptive
python -c "
from src.data.synthetic import generate_transactions
df = generate_transactions(100000)
print(f'Shape: {df.shape}')
print(f'Fraud count: {df.is_fraud.sum()}')
print(f'Fraud ratio: {df.is_fraud.mean():.6f}')
print(df.dtypes)
"
```

### First Commit Message
```
feat(data): add synthetic transaction generator

- Implement generate_transactions() with configurable fraud ratio
- Add realistic fraud patterns (higher amounts, unusual hours)
- Include timestamp generation with fraud-specific distribution
- Set up reproducibility via random_state parameter
```

---

## Quick Reference

### Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/

# Start API (Phase 6)
uvicorn src.api.main:app --reload
```

### Key Files by Phase

| Phase | Primary Files |
|-------|---------------|
| 0 | `src/data/synthetic.py`, `src/data/schemas.py` |
| 1 | `src/features/velocity.py`, `src/features/graph.py`, `src/features/pipeline.py` |
| 2 | `src/models/isolation_forest.py`, `src/models/unsupervised_ensemble.py` |
| 3 | `src/models/custom_loss.py`, `src/models/xgboost_classifier.py` |
| 4 | `src/scoring/threshold_optimizer.py`, `src/scoring/scorer.py` |
| 5 | `src/drift/detector.py`, `src/drift/monitor.py` |
| 6 | `src/api/main.py`, `src/models/registry.py` |
