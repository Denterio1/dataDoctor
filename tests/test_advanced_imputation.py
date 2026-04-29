import pytest
import pandas as pd
import numpy as np
from src.data.advanced_imputer import (
    MissingPatternAnalyzer,
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
    SmartImputer,
    NeuralTabularImputer,
    DenoisingDiffusionImputer,
    TORCH_OK
)

@pytest.fixture
def sample_df_missing():
    """Create a sample dataframe with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'B': [np.nan, 1, 2, 3, np.nan, 5, 6, 7, 8, 9],
        'C': ['cat', 'dog', 'cat', np.nan, 'dog', 'cat', 'dog', 'cat', np.nan, 'dog']
    })
    return df

def test_missing_pattern_analyzer(sample_df_missing):
    analyzer = MissingPatternAnalyzer()
    pattern = analyzer.analyze(sample_df_missing)
    assert pattern.n_rows == 10
    assert pattern.n_cols == 3
    assert pattern.total_missing == 6
    assert 'A' in pattern.col_missing
    assert pattern.mechanism in ["MCAR", "MAR", "NONE"]

def test_simple_imputer(sample_df_missing):
    imputer = SimpleImputer(strategy="mean")
    result = imputer.fit_transform(sample_df_missing)
    assert result.success is True
    assert result.df_imputed['A'].isnull().sum() == 0
    assert result.df_imputed['B'].isnull().sum() == 0
    # Categorical should use mode even with mean strategy
    assert result.df_imputed['C'].isnull().sum() == 0

def test_knn_imputer_fallback(sample_df_missing):
    # Test if it works or falls back gracefully
    imputer = KNNImputer(n_neighbors=2)
    result = imputer.fit_transform(sample_df_missing)
    assert result.success is True
    assert result.df_imputed['A'].isnull().sum() == 0

def test_smart_imputer_selection(sample_df_missing):
    imputer = SmartImputer(prefer_speed=True)
    result = imputer.fit_transform(sample_df_missing)
    assert result.success is True
    assert "SmartImputer" in result.method
    assert result.df_imputed.isnull().sum().sum() == 0

def test_neural_imputer_safety_guard(sample_df_missing):
    """Verify that NeuralTabularImputer falls back if torch is missing or works if present."""
    imputer = NeuralTabularImputer(epochs=1)
    result = imputer.fit_transform(sample_df_missing)
    assert result.success is True
    assert result.df_imputed.isnull().sum().sum() == 0
    # If TORCH_OK was False, it should have a warning about falling back
    if not TORCH_OK:
        assert any("fallback" in str(w).lower() for w in result.warnings)

def test_diffusion_imputer_safety_guard(sample_df_missing):
    """Verify that DenoisingDiffusionImputer falls back safely."""
    imputer = DenoisingDiffusionImputer(steps=2, epochs=1)
    result = imputer.fit_transform(sample_df_missing)
    assert result.success is True
    assert result.df_imputed.isnull().sum().sum() == 0

def test_result_tracking(sample_df_missing):
    imputer = SimpleImputer(strategy="constant", fill_value=99)
    result = imputer.fit_transform(sample_df_missing)
    assert result.n_imputed == 6
    assert result.col_changes['A']['n_filled'] == 2
    assert result.col_changes['A']['sample_val'] == 99
