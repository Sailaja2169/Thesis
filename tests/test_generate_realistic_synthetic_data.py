import pytest
import pandas as pd
import os
import sys
from src.data_collection.generate_realistic_synthetic_data import (
    create_synthetic_records,
    generate_correlated_responses,
    main
)
from src.data_collection import survey_constants as C

@pytest.fixture
def synthetic_records():
    """Fixture to generate synthetic records for testing."""
    return create_synthetic_records(num_records=10)

def test_create_synthetic_records_output_type_and_shape(synthetic_records):
    """Test the output type and shape of create_synthetic_records."""
    assert isinstance(synthetic_records, list)
    assert len(synthetic_records) == 10
    assert isinstance(synthetic_records[0], dict)

def test_create_synthetic_records_columns():
    """Test if key columns from each category are present in the synthetic records."""
    record = create_synthetic_records(num_records=1)[0]
    
    # Check a subset of columns from each category
    essential_columns = [
        C.T_PRIOR_TRAINING,           # Training
        C.T_CONFIDENCE,               # Training confidence
        C.D_DIGITAL_LITERACY,         # Demographics
        C.CB_AUTHORITY,               # Cognitive bias
        C.WC_STRESS,                  # Work context
        C.H_SECURITY_SOFTWARE,        # Security habits
        C.P_SUSPICIOUS_EMAILS,        # Phishing experience
        C.B_CLICK_LINK,               # Behavior
        C.B_REPORT_EMAIL              # Behavior
    ]
    
    for col in essential_columns:
        assert col in record, f"Column {col} is missing from the record"
        assert record[col] is not None, f"Column {col} has a None value"

def test_generate_correlated_responses():
    """Test the logic for generating correlated responses."""
    # Create a complete base record with minimum required fields
    base_record = {
        C.D_DIGITAL_LITERACY: "High",
        C.D_JOB_ROLE: "Management",
        C.T_PRIOR_TRAINING: "No",
        C.T_CONFIDENCE: "5 (Strongly Agree)",
        C.T_EFFECTIVENESS: "5 (Strongly Agree)",
        C.WC_WORKLOAD: "Light",
        C.WC_STRESS: "3 (Neutral)",
        C.WC_TIME_PRESSURE: "Sometimes",
        C.WC_MULTITASKING: "Sometimes",
        C.P_SUSPICIOUS_EMAILS: "Daily",
        C.P_PHISHING_VICTIM: "No",
        C.B_CLICK_LINK: "Often",  # This should get overridden
        C.B_REPORT_EMAIL: "Never", # This should get overridden
        C.B_SHARE_INFO: "Sometimes",
        C.H_SECURITY_SOFTWARE: "Basic"
    }
    
    # Test high digital literacy correlation
    result = generate_correlated_responses(base_record.copy())
    assert result[C.T_PRIOR_TRAINING] == "Yes"  # High literacy correlates with training
    
    # Test IT role correlation
    it_record = base_record.copy()
    it_record[C.D_JOB_ROLE] = "IT/Technical"
    result = generate_correlated_responses(it_record)
    assert result[C.T_PRIOR_TRAINING] == "Yes"  # IT role correlates with training
    assert result[C.D_DIGITAL_LITERACY] in ["Moderate", "High", "Very High"]  # IT correlates with higher literacy

def test_main_function_creates_file():
    """Test if the main function successfully creates the output CSV file."""
    output_dir = "tests/temp_data"
    output_path = os.path.join(output_dir, "test_synthetic_data.csv")
    
    # Ensure the directory exists and the file doesn't
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Temporarily override the default output path in the main function
    # and capture the returned dataframe
    df = main(num_records=20, output_path=output_path)
    
    # Check if file was created
    assert os.path.exists(output_path), f"File was not created at {output_path}"
    
    # Check if returned dataframe contains the expected records
    assert df is not None, "Main function should return a dataframe"
    assert len(df) == 20, f"Expected 20 records but got {len(df)}"
    
    # Check if data is valid by reading it back
    df_read = pd.read_csv(output_path)
    assert len(df_read) == 20, f"Expected 20 records in CSV but got {len(df_read)}"
    assert not df_read.empty, "CSV file should not be empty"
    
    # Cleanup - remove the file but keep the directory for other tests
    if os.path.exists(output_path):
        os.remove(output_path)

def test_all_columns_present_in_final_csv():
    """Check if essential columns are present in the final generated file."""
    output_path = "tests/temp_data/full_test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate the data and get the returned dataframe
    df = main(num_records=5, output_path=output_path)
    
    # Check if the returned dataframe has data
    assert df is not None, "Main function should return a dataframe"
    assert len(df) == 5, f"Expected 5 records but got {len(df) if df is not None else 0}"
    
    # Read back the CSV and check columns
    df_read = pd.read_csv(output_path)
    
    # Check for essential columns (one from each section)
    essential_columns = [C.CB_AUTHORITY, C.WC_STRESS, C.T_PRIOR_TRAINING, C.D_DIGITAL_LITERACY,
                        C.H_SECURITY_SOFTWARE, C.P_SUSPICIOUS_EMAILS, C.B_CLICK_LINK]
    for column in essential_columns:
        assert column in df_read.columns, f"Column {column} not in generated CSV"
    
    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)