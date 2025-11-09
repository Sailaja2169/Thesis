"""
Integration tests to ensure different components of the system work together correctly.
"""
import pytest
import pandas as pd
import os
import numpy as np

from src.data_collection.generate_realistic_synthetic_data import main as generate_data
from src.processing.real_data_processor import calculate_vulnerability_score, clean_and_prepare_data

def test_end_to_end_data_flow():
    """
    Test the full data flow from generation to vulnerability scoring.
    This ensures that the data generation and processing components work together.
    """
    # Generate synthetic data
    output_path = "tests/temp_data/integration_test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate 10 synthetic records
    df_generated = generate_data(num_records=10, output_path=output_path)
    
    # Ensure the data was generated and saved correctly
    assert df_generated is not None
    assert len(df_generated) == 10
    assert os.path.exists(output_path)
    
    # Process the generated data
    df_processed = calculate_vulnerability_score(df_generated)
    
    # Check that vulnerability scores were calculated
    assert 'vulnerability_score' in df_processed.columns
    assert not df_processed['vulnerability_score'].isnull().any()
    
    # Check that the scores are normalized correctly
    assert df_processed['vulnerability_score'].min() >= 0
    assert df_processed['vulnerability_score'].max() <= 1
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)

def test_clean_and_prepare_with_synthetic_data():
    """
    Test that the clean_and_prepare_data function works with our synthetic data.
    """
    # Generate synthetic data
    output_path = "tests/temp_data/clean_prepare_test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate 5 synthetic records
    generate_data(num_records=5, output_path=output_path)
    
    # Process the data using the full pipeline
    df_prepared = clean_and_prepare_data(output_path)
    
    # Check essential outputs
    assert df_prepared is not None
    assert 'vulnerability_score' in df_prepared.columns
    assert len(df_prepared) == 5
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)