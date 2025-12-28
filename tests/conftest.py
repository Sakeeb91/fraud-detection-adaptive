"""Shared pytest fixtures for fraud detection tests."""

import pytest
import pandas as pd
from src.data.synthetic import generate_transactions


@pytest.fixture
def sample_transactions():
    """Generate a small sample of transactions for testing."""
    return generate_transactions(n_transactions=1000, fraud_ratio=0.01, random_state=42)


@pytest.fixture
def large_transactions():
    """Generate a larger dataset with realistic fraud ratio."""
    return generate_transactions(n_transactions=100000, fraud_ratio=0.0001, random_state=42)
