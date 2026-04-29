import pytest
import pandas as pd
import numpy as np
from src.data.advanced_imputer import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
    SmartImputer,
    MissingPatternAnalyzer,
    NeuralTabularImputer,
    AdvancedMultimodalImputer,
    TORCH_OK
)

@pytest.fixture
def sample_df():
    """Create a sample dataframe with mixed types and missing values."""
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['cat', 'dog', 'cat', np.nan, 'dog'],
        'D': [10, 20, 30, 40, 50]
    })

@pytest.fixture
def numeric_df():
    """Create a numeric-only dataframe with missing values."""
    return pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [np.nan, 2.2, 3.3, 4.4, 5.5],
        'C': [10, 20, 30, 40, 50]
    })

def test_missing_pattern_analyzer(sample_df):
    analyzer = MissingPatternAnalyzer()
    pattern = analyzer.analyze(sample_df)
    assert pattern.n_rows == 5
    assert pattern.total_missing == 4
    assert 'A' in pattern.col_missing

def test_simple_imputer_strategies(sample_df):
    imp = SimpleImputer(strategy="mean")
    res = imp.fit_transform(sample_df)
    assert res.success
    assert not res.df_imputed['A'].isnull().any()
    # Categorical 'C' should use mode as fallback for mean
    assert not res.df_imputed['C'].isnull().any()

def test_knn_imputer(sample_df):
    imp = KNNImputer(n_neighbors=2)
    res = imp.fit_transform(sample_df)
    assert res.success
    assert not res.df_imputed.isnull().any().any()

def test_smart_imputer_auto_selection(sample_df):
    imp = SmartImputer(prefer_speed=True)
    res = imp.fit_transform(sample_df)
    assert res.success
    assert not res.df_imputed.isnull().any().any()

def test_neural_imputer_numeric_only(numeric_df):
    """Neural imputer specifically targets numeric columns."""
    imp = NeuralTabularImputer(epochs=1)
    res = imp.fit_transform(numeric_df)
    assert res.success
    assert not res.df_imputed.isnull().any().any()
    
    if not TORCH_OK:
        assert any("PyTorch not installed" in w for w in res.warnings)

def test_advanced_multimodal_orchestrator(sample_df):
    """Verify that the multimodal pipeline handles mixed types."""
    imp = AdvancedMultimodalImputer(use_text_ai=False, use_neural=False)
    res = imp.fit_transform(sample_df)
    assert res.success
    assert not res.df_imputed.isnull().any().any()

def test_imputation_result_tracking(sample_df):
    imp = SimpleImputer(strategy="constant", fill_value=99)
    res = imp.fit_transform(sample_df)
    assert res.n_imputed == 4
    assert 'A' in res.col_changes
    assert res.col_changes['A']['n_filled'] == 1
    assert res.col_changes['A']['sample_val'] == 99
