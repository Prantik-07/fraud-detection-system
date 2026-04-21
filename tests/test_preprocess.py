"""
tests/test_preprocess.py — Unit tests for the preprocessing pipeline.

Run: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import scale_features, split_data, apply_smote


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a small synthetic dataset mimicking creditcard.csv structure."""
    np.random.seed(42)
    n = 1000
    v_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame({
        "Time":  np.random.uniform(0, 172800, n),
        **v_cols,
        "Amount": np.random.exponential(scale=88, size=n),
        "Class": np.where(np.random.rand(n) < 0.02, 1, 0),  # ~2% fraud
    })
    return df


@pytest.fixture
def split_data_fixture(sample_df):
    """Run scale + split, return all components."""
    df_scaled, scaler = scale_features(sample_df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df_scaled, test_size=0.2)
    return X_train, X_test, y_train, y_test, scaler


# ── Tests: scale_features ──────────────────────────────────────────────────────
class TestScaleFeatures:

    def test_output_shape_unchanged(self, sample_df):
        """Scaling should not change the shape of the DataFrame."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        assert df_scaled.shape == sample_df.shape

    def test_amount_time_scaled(self, sample_df):
        """Amount and Time columns should have near-zero mean after scaling."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        assert abs(df_scaled["Amount"].mean()) < 1.0, "Amount mean should be near 0"
        assert abs(df_scaled["Time"].mean()) < 1.0,   "Time mean should be near 0"

    def test_v_columns_unchanged(self, sample_df):
        """V1–V28 columns must NOT be modified by the scaler."""
        original_v1 = sample_df["V1"].values.copy()
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        np.testing.assert_array_almost_equal(df_scaled["V1"].values, original_v1)

    def test_class_column_unchanged(self, sample_df):
        """Class labels must not be altered during scaling."""
        original_class = sample_df["Class"].values.copy()
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        np.testing.assert_array_equal(df_scaled["Class"].values, original_class)

    def test_scaler_returned(self, sample_df):
        """A fitted scaler object must be returned."""
        from sklearn.preprocessing import StandardScaler
        _, scaler = scale_features(sample_df.copy(), fit=True)
        assert isinstance(scaler, StandardScaler)


# ── Tests: split_data ─────────────────────────────────────────────────────────
class TestSplitData:

    def test_split_sizes(self, sample_df):
        """80/20 split should produce approximately correct sizes."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        X_train, X_test, y_train, y_test = split_data(df_scaled, test_size=0.2)

        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.2) < 0.02, "Test size should be ~20%"

    def test_no_overlap(self, sample_df):
        """Train and test sets must not share any rows."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        X_train, X_test, _, _ = split_data(df_scaled, test_size=0.2)

        # Check that no row in X_test exists in X_train
        train_set = set(map(tuple, X_train.tolist()))
        test_set  = set(map(tuple, X_test.tolist()))
        overlap   = train_set & test_set
        assert len(overlap) == 0, "Train and test sets must not overlap"

    def test_stratification(self, sample_df):
        """Stratified split should preserve the fraud ratio in both sets."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        X_train, X_test, y_train, y_test = split_data(df_scaled, test_size=0.2)

        overall_rate = sample_df["Class"].mean()
        train_rate   = y_train.mean()
        test_rate    = y_test.mean()

        assert abs(train_rate - overall_rate) < 0.05, "Train fraud rate should match overall"
        assert abs(test_rate  - overall_rate) < 0.05, "Test fraud rate should match overall"

    def test_output_types(self, sample_df):
        """Outputs should be NumPy arrays."""
        df_scaled, _ = scale_features(sample_df.copy(), fit=True)
        X_train, X_test, y_train, y_test = split_data(df_scaled)
        for arr in [X_train, X_test, y_train, y_test]:
            assert isinstance(arr, np.ndarray), "All split outputs must be NumPy arrays"


# ── Tests: apply_smote ────────────────────────────────────────────────────────
class TestSMOTE:

    def test_smote_balances_classes(self, split_data_fixture):
        """After SMOTE, classes should be approximately balanced."""
        X_train, _, y_train, _, _ = split_data_fixture
        X_res, y_res = apply_smote(X_train, y_train)
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1], "SMOTE should produce equal class counts"

    def test_smote_increases_minority(self, split_data_fixture):
        """SMOTE must increase the minority class count."""
        X_train, _, y_train, _, _ = split_data_fixture
        fraud_before = y_train.sum()
        _, y_res = apply_smote(X_train, y_train)
        fraud_after = y_res.sum()
        assert fraud_after > fraud_before, "SMOTE must create more minority samples"

    def test_smote_not_on_test_set(self, split_data_fixture):
        """Test: confirm SMOTE only receives train data (test labels are unchanged)."""
        X_train, X_test, y_train, y_test, _ = split_data_fixture
        original_test_sum = y_test.sum()
        # SMOTE applied to train only — test must remain the same
        _, _ = apply_smote(X_train, y_train)
        assert y_test.sum() == original_test_sum, "SMOTE must NOT touch test labels"

    def test_smote_output_shape(self, split_data_fixture):
        """SMOTE output must have correct number of features."""
        X_train, _, y_train, _, _ = split_data_fixture
        X_res, y_res = apply_smote(X_train, y_train)
        assert X_res.shape[1] == X_train.shape[1], "Feature count must not change after SMOTE"
        assert len(X_res) == len(y_res), "X and y must have same length after SMOTE"


# ── Tests: model loading ───────────────────────────────────────────────────────
class TestModelLoading:

    def test_model_files_exist(self):
        """Trained model files should exist (run src.train first)."""
        expected = [
            os.path.join("models", "best_model.pkl"),
            os.path.join("models", "scaler.pkl"),
            os.path.join("models", "metrics.json"),
        ]
        for path in expected:
            if not os.path.exists(path):
                pytest.skip(f"{path} not found — run `python -m src.train` first")
            assert os.path.exists(path), f"Expected file not found: {path}"

    def test_model_can_predict(self):
        """Loaded model must be callable for predictions."""
        import joblib
        model_path = os.path.join("models", "best_model.pkl")
        if not os.path.exists(model_path):
            pytest.skip("Model not trained yet")
        model = joblib.load(model_path)
        X_dummy = np.zeros((1, 30))
        pred = model.predict(X_dummy)
        assert pred[0] in [0, 1], "Prediction must be 0 or 1"
