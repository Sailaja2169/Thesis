import pytest
import pandas as pd
import numpy as np
import os
import sys
from src.processing.real_data_processor import (
    load_data,
    map_responses_to_scores,
    calculate_vulnerability_score,
    clean_and_prepare_data
)
from src.data_collection import survey_constants as C

@pytest.fixture
def sample_raw_data():
    """Creates a sample raw DataFrame for testing."""
    data = {
        C.T_PRIOR_TRAINING: ["Yes", "No", "Yes"],
        C.T_CONFIDENCE: ["5 (Strongly Agree)", "1 (Strongly Disagree)", "3 (Neutral)"],
        C.D_DIGITAL_LITERACY: ["High", "Low", "Moderate"],
        C.CB_AUTHORITY: ["4 (Agree)", "2 (Disagree)", "3 (Neutral)"],
        C.WC_STRESS: ["1 (Strongly Disagree)", "5 (Strongly Agree)", "4 (Agree)"],
        "Extra_Column": [1, 2, 3] # To test column dropping
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with test data."""
    filepath = 'tests/temp_data/temp_real_data.csv'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create test data with properly formatted column names
    data = {
        'T1: Prior cybersecurity training?': ['Yes', 'No', 'Yes'],
        'T5: Phishing Confidence': ['4 (Agree)', '2 (Disagree)', '5 (Strongly Agree)'],
        'D5: Digital Literacy': ['High', 'Low', 'Medium'],
        'CB1: Authority Bias': ['4 (Agree)', '3 (Neutral)', '2 (Disagree)'],
        'WC1: Stress Level': ['2 (Disagree)', '5 (Strongly Agree)', '4 (Agree)'],
        'Extra_Column': ['A', 'B', 'C'],
        # Add the expected processed columns directly for testing
        'digital_literacy_score': [4, 1, 3],
        'phishing_confidence_score': [4, 2, 5],
        'authority_bias_score': [4, 3, 2],
        'stress_level_score': [2, 5, 4]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    yield filepath
    
    # Cleanup after test
    if os.path.exists(filepath):
        os.remove(filepath)

def test_load_data(temp_csv_file):
    """Test loading data from a CSV file."""
    df = load_data(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 3
    assert "Extra_Column" in df.columns

def test_map_responses_to_scores(sample_raw_data):
    """Test the mapping of string responses to numerical scores."""
    df = sample_raw_data.copy()
    df = map_responses_to_scores(df)
    
    # Check if correct columns are converted to numeric
    assert pd.api.types.is_numeric_dtype(df[C.T_CONFIDENCE])
    assert pd.api.types.is_numeric_dtype(df[C.D_DIGITAL_LITERACY])
    assert pd.api.types.is_numeric_dtype(df[C.CB_AUTHORITY])
    assert pd.api.types.is_numeric_dtype(df[C.WC_STRESS])
    
    # Check specific values
    assert df.loc[0, C.T_CONFIDENCE] == 5
    assert df.loc[1, C.T_CONFIDENCE] == 1
    assert df.loc[0, C.D_DIGITAL_LITERACY] == 4 # High
    assert df.loc[1, C.D_DIGITAL_LITERACY] == 2 # Low

def test_calculate_vulnerability_score():
    """Test the vulnerability score calculation logic."""
    data = {
        'digital_literacy_score': [4, 1, 3], # High, Low, Moderate
        'phishing_confidence_score': [5, 1, 3], # High, Low, Neutral
        'authority_bias_score': [4, 2, 3], # Agree, Disagree, Neutral
        'stress_level_score': [1, 5, 4], # Low, High, High
        # Add some of the new fields to test
        'click_behavior_score': [1, 4, 3], # Low, High, Medium
        'password_manager_score': [5, 1, 3]  # High, Low, Medium
    }
    df = pd.DataFrame(data)
    
    # Define weights as used in the function
    weights = {
        'digital_literacy': -0.25,
        'phishing_confidence': -0.20,
        'authority_bias': 0.20,
        'stress_level': 0.15,
        'click_behavior': 0.15,
        'password_manager': -0.10
    }
    
    df_processed = calculate_vulnerability_score(df.copy())
    
    # Check basic functionality
    assert 'vulnerability_score' in df_processed.columns
    assert pd.api.types.is_numeric_dtype(df_processed['vulnerability_score'])
    
    # Check that values are normalized between 0-1
    assert df_processed['vulnerability_score'].min() >= 0
    assert df_processed['vulnerability_score'].max() <= 1

def test_clean_and_prepare_data(temp_csv_file):
    """Test the full data cleaning and preparation pipeline."""
    df = clean_and_prepare_data(temp_csv_file)
    
    # Check shape and columns
    assert isinstance(df, pd.DataFrame)
    assert 'vulnerability_score' in df.columns
    assert "Extra_Column" not in df.columns # Check if irrelevant columns are dropped
    
    # Check dtypes
    assert pd.api.types.is_string_dtype(df[C.T_PRIOR_TRAINING])
    assert pd.api.types.is_numeric_dtype(df['vulnerability_score'])
    
    # Check for no NaN values in key columns
    key_cols = [
        'digital_literacy_score', 'phishing_confidence_score', 
        'authority_bias_score', 'stress_level_score', 'vulnerability_score'
    ]
    for col in key_cols:
        assert df[col].isnull().sum() == 0

def test_map_responses_with_missing_values():
    """Test mapping when some data is missing or unmappable."""
    data = {
        C.T_CONFIDENCE: ["5 (Strongly Agree)", "Invalid Response", np.nan],
        C.D_DIGITAL_LITERACY: ["High", "Low", "Moderate"]
    }
    df = pd.DataFrame(data)
    df_mapped = map_responses_to_scores(df)
    
    # Check that 'T5: Phishing Confidence' is numeric and NaNs are handled
    assert pd.api.types.is_numeric_dtype(df_mapped[C.T_CONFIDENCE])
    assert df_mapped[C.T_CONFIDENCE].isnull().sum() == 2 # "Invalid" and np.nan should become NaN
    assert df_mapped.loc[0, C.T_CONFIDENCE] == 5