"""
This script generates a realistic synthetic dataset of 300 survey responses
for the phishing vulnerability research project. It creates logical correlations
between different survey answers to provide a richer, more life-like dataset
for analysis when real-world data collection is pending.

Key Features:
- Generates 300 records.
- Uses Faker to create realistic names, emails, and timestamps.
- Employs weighted distributions for demographic and psychometric data.
- Enforces logical correlations (e.g., IT staff have more training).
- Exports the data to a CSV file in the `data/raw/` directory.
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
import os
import sys
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to allow importing constants
# This makes the script runnable from any directory
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.data_collection import survey_constants as C
except ImportError as e:
    logging.error(f"Error importing survey_constants: {e}")
    logging.error("Please ensure the script is run from the project's root directory or the path is correctly set.")
    sys.exit(1)


# --- Configuration ---
NUM_RECORDS = 300
OUTPUT_PATH = "data/raw/synthetic_survey_data_realistic_300.csv"

# Initialize Faker for generating realistic text data
fake = Faker()

# --- Define Realistic Choices and Distributions ---
age_dist = {"18-24": 0.15, "25-34": 0.35, "35-44": 0.25, "45-54": 0.15, "55+": 0.10}
education_dist = {"High School": 0.10, "Bachelor's Degree": 0.50, "Master's Degree": 0.35, "PhD": 0.05}
job_role_dist = {"IT/Technical": 0.25, "Management": 0.20, "Sales/Marketing": 0.15, "HR/Admin": 0.15, "Finance": 0.10, "Other": 0.15}
experience_dist = {"0-2 years": 0.15, "3-5 years": 0.25, "6-10 years": 0.30, "11-20 years": 0.20, "20+ years": 0.10}
digital_literacy_dist = {"Very Low": 0.05, "Low": 0.10, "Moderate": 0.40, "High": 0.35, "Very High": 0.10}
likert_scale_dist = {"1 (Strongly Disagree)": 0.1, "2 (Disagree)": 0.15, "3 (Neutral)": 0.25, "4 (Agree)": 0.35, "5 (Strongly Agree)": 0.15}
frequency_dist = {"Never": 0.05, "Rarely": 0.20, "Sometimes": 0.40, "Often": 0.25, "Always": 0.10}
training_recency_dist = {"Within the last 6 months": 0.4, "6-12 months ago": 0.3, "1-2 years ago": 0.2, "More than 2 years ago": 0.1}
training_type_dist = {"Online modules": 0.5, "In-person workshop": 0.2, "Phishing simulations": 0.2, "Informal (e.g., reading articles)": 0.1}
training_duration_dist = {"Less than 1 hour": 0.3, "1-2 hours": 0.5, "Half-day": 0.15, "Full-day or more": 0.05}
email_volume_dist = {"Less than 20": 0.15, "20-50": 0.40, "51-100": 0.30, "101-200": 0.10, "More than 200": 0.05}
workload_dist = {"Very Light": 0.05, "Light": 0.20, "Manageable": 0.50, "Heavy": 0.20, "Very Heavy": 0.05}

def get_weighted_choice(distribution):
    """
    Selects an item from a dictionary-based distribution.

    Args:
        distribution (dict): A dictionary where keys are choices and values are their weights.

    Returns:
        The selected key based on the weighted random choice.
    """
    return random.choices(list(distribution.keys()), weights=list(distribution.values()), k=1)[0]

def generate_correlated_responses(record):
    """
    Generates logical correlations between different fields in a survey record.
    This function introduces realistic dependencies between answers to make the
    synthetic data more life-like.

    Args:
        record (dict): A dictionary representing a single survey response.

    Returns:
        dict: The modified record with logical correlations applied.
    """
    # High digital literacy correlates with better training and confidence
    if record[C.D_DIGITAL_LITERACY] in ["High", "Very High"]:
        record[C.T_PRIOR_TRAINING] = "Yes"
        record[C.T_CONFIDENCE] = get_weighted_choice({"3 (Neutral)": 0.1, "4 (Agree)": 0.5, "5 (Strongly Agree)": 0.4})
        record[C.T_EFFECTIVENESS] = get_weighted_choice({"3 (Neutral)": 0.1, "4 (Agree)": 0.5, "5 (Strongly Agree)": 0.4})
        # High digital literacy also correlates with better security practices
        record[C.H_SECURITY_SOFTWARE] = get_weighted_choice({"Standard": 0.3, "Advanced": 0.7})
        record[C.H_PASSWORD_MANAGER] = get_weighted_choice({"Sometimes": 0.2, "Often": 0.4, "Always": 0.4})
        record[C.H_UPDATE_FREQUENCY] = get_weighted_choice({"Monthly": 0.3, "Weekly": 0.5, "Daily": 0.2})
        record[C.B_REPORT_EMAIL] = get_weighted_choice({"Sometimes": 0.3, "Often": 0.4, "Always": 0.3})

    # IT roles tend to have more training and higher literacy
    if record[C.D_JOB_ROLE] == "IT/Technical":
        record[C.D_DIGITAL_LITERACY] = get_weighted_choice({"Moderate": 0.1, "High": 0.5, "Very High": 0.4})
        record[C.T_PRIOR_TRAINING] = "Yes"
        record[C.T_RECENCY] = get_weighted_choice({"Within the last 6 months": 0.6, "6-12 months ago": 0.4})
        # IT roles are more likely to have advanced security practices
        record[C.H_SECURITY_SOFTWARE] = get_weighted_choice({"Standard": 0.2, "Advanced": 0.8})
        record[C.H_PASSWORD_MANAGER] = get_weighted_choice({"Often": 0.4, "Always": 0.6})
        record[C.P_REPORTING] = get_weighted_choice({"Sometimes": 0.2, "Always": 0.8})
        record[C.B_CLICK_LINK] = get_weighted_choice({"Never": 0.5, "Rarely": 0.5})

    # Heavy workload correlates with higher stress and time pressure
    if record[C.WC_WORKLOAD] in ["Heavy", "Very Heavy"]:
        record[C.WC_STRESS] = get_weighted_choice({"3 (Neutral)": 0.1, "4 (Agree)": 0.5, "5 (Strongly Agree)": 0.4})
        record[C.WC_TIME_PRESSURE] = get_weighted_choice({"Sometimes": 0.1, "Often": 0.5, "Always": 0.4})
        record[C.WC_MULTITASKING] = get_weighted_choice({"Sometimes": 0.1, "Often": 0.5, "Always": 0.4})
        # People with heavy workloads are more likely to take shortcuts
        record[C.H_UPDATE_FREQUENCY] = get_weighted_choice({"Never": 0.1, "When prompted": 0.7, "Monthly": 0.2})
        record[C.B_CLICK_LINK] = get_weighted_choice({"Sometimes": 0.4, "Often": 0.4, "Always": 0.2})

    # If no training, set training-related fields to N/A
    if record[C.T_PRIOR_TRAINING] == "No":
        record[C.T_RECENCY] = "N/A"
        record[C.T_TYPE] = "N/A"
        record[C.T_DURATION] = "N/A"
        record[C.T_EFFECTIVENESS] = "N/A"
        record[C.T_CONFIDENCE] = get_weighted_choice({"1 (Strongly Disagree)": 0.4, "2 (Disagree)": 0.5, "3 (Neutral)": 0.1})
        # Less training correlates with poorer security behaviors
        record[C.B_CLICK_LINK] = get_weighted_choice({"Sometimes": 0.3, "Often": 0.4, "Always": 0.3})
        record[C.B_REPORT_EMAIL] = get_weighted_choice({"Never": 0.4, "Rarely": 0.4, "Sometimes": 0.2})
    
    # Previous phishing victims are more likely to be cautious
    if record[C.P_PHISHING_VICTIM] in ["Yes, once", "Yes, multiple times"]:
        record[C.B_CLICK_LINK] = get_weighted_choice({"Never": 0.3, "Rarely": 0.5, "Sometimes": 0.2})
        record[C.P_REPORTING] = get_weighted_choice({"Sometimes": 0.3, "Always": 0.7})
        record[C.H_SECURITY_SOFTWARE] = get_weighted_choice({"Basic": 0.3, "Standard": 0.5, "Advanced": 0.2})
    
    # People who receive frequent suspicious emails are more aware
    if record[C.P_SUSPICIOUS_EMAILS] == "Daily":
        record[C.P_PHISHING_VICTIM] = get_weighted_choice({"No": 0.5, "Yes, once": 0.3, "Yes, multiple times": 0.2})
        record[C.T_CONFIDENCE] = get_weighted_choice({"3 (Neutral)": 0.2, "4 (Agree)": 0.5, "5 (Strongly Agree)": 0.3})

    return record

def create_synthetic_records(num_records):
    """
    Generates a specified number of synthetic survey records.

    Args:
        num_records (int): The number of records to generate.

    Returns:
        list: A list of dictionaries, where each dictionary is a survey record.
    """
    # Define new distributions for security habits
    security_software_dist = {"None": 0.1, "Basic": 0.3, "Standard": 0.4, "Advanced": 0.2}
    password_manager_dist = {"Never": 0.3, "Rarely": 0.2, "Sometimes": 0.2, "Often": 0.2, "Always": 0.1}
    update_frequency_dist = {"Never": 0.05, "When prompted": 0.4, "Monthly": 0.3, "Weekly": 0.2, "Daily": 0.05}
    backup_frequency_dist = {"Never": 0.1, "Yearly": 0.2, "Monthly": 0.4, "Weekly": 0.2, "Daily": 0.1}
    
    # Define phishing experience distributions
    suspicious_emails_dist = {"Never": 0.05, "Monthly": 0.2, "Weekly": 0.5, "Daily": 0.25}
    phishing_victim_dist = {"No": 0.7, "Yes, once": 0.2, "Yes, multiple times": 0.1}
    reporting_dist = {"Never": 0.2, "Rarely": 0.3, "Sometimes": 0.3, "Always": 0.2}
    
    # Define behavior distributions
    behavior_dist = {"Never": 0.1, "Rarely": 0.3, "Sometimes": 0.3, "Often": 0.2, "Always": 0.1}
    
    records = []
    for i in range(num_records):
        if (i + 1) % 50 == 0:
            logging.info(f"Generating record {i+1} of {num_records}...")
        record = {
            C.TIMESTAMP: fake.date_time_this_year().strftime('%Y/%m/%d %H:%M:%S'),
            C.EMAIL_ADDRESS: fake.email(),
            C.CONSENT: "I consent",
            C.CB_AUTHORITY: get_weighted_choice(likert_scale_dist),
            C.CB_URGENCY: get_weighted_choice(likert_scale_dist),
            C.CB_FAMILIARITY: get_weighted_choice(likert_scale_dist),
            C.CB_CURIOSITY: get_weighted_choice(likert_scale_dist),
            C.WC_STRESS: get_weighted_choice(likert_scale_dist),
            C.WC_MULTITASKING: get_weighted_choice(frequency_dist),
            C.WC_EMAIL_VOLUME: get_weighted_choice(email_volume_dist),
            C.WC_WORKLOAD: get_weighted_choice(workload_dist),
            C.WC_TIME_PRESSURE: get_weighted_choice(frequency_dist),
            C.T_PRIOR_TRAINING: random.choices(["Yes", "No"], weights=[0.7, 0.3], k=1)[0],
            C.T_RECENCY: get_weighted_choice(training_recency_dist),
            C.T_TYPE: get_weighted_choice(training_type_dist),
            C.T_DURATION: get_weighted_choice(training_duration_dist),
            C.T_CONFIDENCE: get_weighted_choice(likert_scale_dist),
            C.T_EFFECTIVENESS: get_weighted_choice(likert_scale_dist),
            C.D_AGE: get_weighted_choice(age_dist),
            C.D_EDUCATION: get_weighted_choice(education_dist),
            C.D_JOB_ROLE: get_weighted_choice(job_role_dist),
            C.D_EXPERIENCE: get_weighted_choice(experience_dist),
            C.D_DIGITAL_LITERACY: get_weighted_choice(digital_literacy_dist),
            # Add new security habits fields
            C.H_SECURITY_SOFTWARE: get_weighted_choice(security_software_dist),
            C.H_PASSWORD_MANAGER: get_weighted_choice(password_manager_dist),
            C.H_UPDATE_FREQUENCY: get_weighted_choice(update_frequency_dist),
            C.H_BACKUP_FREQUENCY: get_weighted_choice(backup_frequency_dist),
            # Add phishing experience fields
            C.P_SUSPICIOUS_EMAILS: get_weighted_choice(suspicious_emails_dist),
            C.P_PHISHING_VICTIM: get_weighted_choice(phishing_victim_dist),
            C.P_REPORTING: get_weighted_choice(reporting_dist),
            # Add behavior fields
            C.B_CLICK_LINK: get_weighted_choice(behavior_dist),
            C.B_REPORT_EMAIL: get_weighted_choice(behavior_dist),
            C.B_SHARE_INFO: get_weighted_choice(behavior_dist),
        }
        # Apply logical correlations
        record = generate_correlated_responses(record)
        records.append(record)
    return records

def main(num_records=NUM_RECORDS, output_path=OUTPUT_PATH):
    """
    Main function to generate and save the dataset.
    
    Args:
        num_records (int, optional): Number of records to generate. Defaults to NUM_RECORDS.
        output_path (str, optional): Path to save the CSV file. Defaults to OUTPUT_PATH.
        
    Returns:
        pandas.DataFrame: The generated dataset, or None if an error occurred.
    """
    logging.info(f"Starting generation of {num_records} realistic synthetic records...")
    try:
        synthetic_data = create_synthetic_records(num_records)
        df = pd.DataFrame(synthetic_data)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logging.info(f"Creating directory: {output_dir}")
            os.makedirs(output_dir)
        
        df.to_csv(output_path, index=False)
        logging.info(f"✅ Successfully generated and saved data to {output_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f"❌ File not found error: {e}. Check if the path is correct.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}")
    return None
        
if __name__ == "__main__":
    main()