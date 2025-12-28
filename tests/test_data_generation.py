"""Tests for synthetic data generation."""

import pytest
import pandas as pd
import numpy as np
from src.data.synthetic import generate_transactions


class TestGenerateTransactions:
    """Test suite for generate_transactions function."""

    def test_output_shape(self):
        """Test that output has expected number of rows and columns."""
        df = generate_transactions(1000)
        assert len(df) == 1000
        assert len(df.columns) == 6

    def test_expected_columns(self):
        """Test that output has all expected columns."""
        df = generate_transactions(100)
        expected_columns = {'transaction_id', 'timestamp', 'user_id',
                          'merchant_id', 'amount', 'is_fraud'}
        assert set(df.columns) == expected_columns

    def test_fraud_ratio_approximate(self):
        """Test that fraud ratio is approximately as specified."""
        df = generate_transactions(100000, fraud_ratio=0.0001)
        actual_ratio = df['is_fraud'].mean()
        # Within 50% of target for large samples
        assert 0.00005 < actual_ratio < 0.00015

    def test_fraud_ratio_custom(self):
        """Test custom fraud ratio."""
        df = generate_transactions(10000, fraud_ratio=0.01)
        actual_ratio = df['is_fraud'].mean()
        assert 0.008 < actual_ratio < 0.012

    def test_reproducibility(self):
        """Test that same random_state produces same output."""
        df1 = generate_transactions(100, random_state=42)
        df2 = generate_transactions(100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        df1 = generate_transactions(100, random_state=42)
        df2 = generate_transactions(100, random_state=123)
        assert not df1.equals(df2)

    def test_column_types(self):
        """Test that columns have correct data types."""
        df = generate_transactions(100)
        assert df['transaction_id'].dtype == object  # string
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert df['user_id'].dtype == object
        assert df['merchant_id'].dtype == object
        assert df['amount'].dtype == float
        assert df['is_fraud'].dtype == int

    def test_no_nan_values(self):
        """Test that there are no NaN values."""
        df = generate_transactions(1000)
        assert not df.isna().any().any()

    def test_amounts_positive(self):
        """Test that all amounts are positive."""
        df = generate_transactions(1000)
        assert (df['amount'] > 0).all()

    def test_is_fraud_binary(self):
        """Test that is_fraud contains only 0 and 1."""
        df = generate_transactions(1000, fraud_ratio=0.1)
        assert set(df['is_fraud'].unique()).issubset({0, 1})

    def test_minimum_one_fraud(self):
        """Test that there's at least one fraud even with tiny ratio."""
        df = generate_transactions(100, fraud_ratio=0.0001)
        assert df['is_fraud'].sum() >= 1

    def test_fraud_amounts_higher(self):
        """Test that fraud transactions have higher average amount."""
        df = generate_transactions(10000, fraud_ratio=0.1)
        fraud_mean = df[df['is_fraud'] == 1]['amount'].mean()
        legit_mean = df[df['is_fraud'] == 0]['amount'].mean()
        assert fraud_mean > legit_mean * 2  # At least 2x higher

    def test_transaction_ids_unique(self):
        """Test that transaction IDs are unique."""
        df = generate_transactions(1000)
        assert df['transaction_id'].nunique() == len(df)


class TestTimestampGeneration:
    """Test suite for timestamp generation."""

    def test_timestamps_in_range(self):
        """Test that timestamps are within expected range."""
        df = generate_transactions(1000)
        min_date = pd.Timestamp('2024-01-01')
        max_date = pd.Timestamp('2024-12-31')
        assert (df['timestamp'] >= min_date).all()
        assert (df['timestamp'] <= max_date).all()

    def test_fraud_unusual_hours(self):
        """Test that fraud transactions tend to occur at unusual hours."""
        df = generate_transactions(10000, fraud_ratio=0.1)
        fraud_hours = df[df['is_fraud'] == 1]['timestamp'].dt.hour
        legit_hours = df[df['is_fraud'] == 0]['timestamp'].dt.hour

        # Fraud should have more transactions in 2-5 AM range
        fraud_unusual = ((fraud_hours >= 2) & (fraud_hours <= 5)).mean()
        legit_unusual = ((legit_hours >= 2) & (legit_hours <= 5)).mean()

        assert fraud_unusual > legit_unusual
