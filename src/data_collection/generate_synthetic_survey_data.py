"""
Synthetic Data Generator for Phishing Survey
Creates realistic fake data based on actual survey structure
Generates 400+ responses over last 20 days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Survey configuration
NUM_RESPONSES = 420  # Generate 420 responses
DAYS_BACK = 20  # Distributed over last 20 days

# Define response options based on actual survey
AGE_BRACKETS = ['18–25', '26–35', '36–45', '46–55', '56–65', '65+']
YEARS_EXPERIENCE = ['< 1', '1 - 3', '3 - 5', '5+']
TRAINING_OPTIONS = ['Yes', 'No']
JOB_ROLES = [
    'Software Developer', 'Software Engineer', 'Data Analyst', 'IT Support',
    'Project Manager', 'Business Analyst', 'System Administrator', 
    'Network Engineer', 'Database Administrator', 'DevOps Engineer',
    'Finance Manager', 'HR Manager', 'Marketing Manager', 'Sales Manager',
    'Administrative Assistant', 'Accountant', 'Customer Support',
    'Research Scientist', 'Teacher', 'Healthcare Professional',
    'Legal Advisor', 'Operations Manager', 'Quality Assurance'
]
MULTITASK_FREQ = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']
CONSENT_OPTIONS = ['I Consent', 'I do not consent']

# Probability distributions (realistic patterns)
AGE_PROBS = [0.25, 0.35, 0.20, 0.12, 0.06, 0.02]  # Younger workforce more represented
EXPERIENCE_PROBS = [0.08, 0.22, 0.25, 0.45]  # More experienced users
TRAINING_PROBS = [0.62, 0.38]  # 62% have training (realistic)
MULTITASK_PROBS = [0.05, 0.12, 0.25, 0.38, 0.20]  # Most people multitask often


def generate_timestamp(days_back, response_index, total_responses):
    """
    Generate realistic timestamps distributed over last N days
    More responses in recent days (realistic collection pattern)
    """
    end_date = datetime(2025, 10, 2, 23, 59, 59)  # Today
    start_date = end_date - timedelta(days=days_back)
    
    # Weight towards recent days (exponential decay)
    # More responses collected towards the end of the period
    progress = response_index / total_responses
    weighted_progress = progress ** 0.5  # Square root makes it more gradual
    
    time_span = (end_date - start_date).total_seconds()
    random_offset = random.uniform(0, time_span * 0.1)  # Add some randomness
    
    timestamp = start_date + timedelta(seconds=time_span * weighted_progress + random_offset)
    
    # Add some time of day variation (most responses during work hours 9am-6pm)
    hour = random.choices(
        range(24),
        weights=[1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 10, 10, 9, 9, 8, 7, 6, 5, 3, 2, 2, 2, 1, 1]
    )[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    timestamp = timestamp.replace(hour=hour, minute=minute, second=second)
    
    return timestamp.strftime('%d/%m/%Y %H:%M:%S')


def generate_correlated_scores(base_score, correlation=0.6):
    """
    Generate correlated scores (people's responses tend to be consistent)
    """
    noise = np.random.normal(0, 1 - correlation)
    score = base_score + noise
    return max(1, min(5, int(round(score))))


def generate_realistic_response(participant_id, timestamp, real_patterns):
    """
    Generate one realistic survey response with correlations
    """
    # Determine if this person will consent (95% consent rate)
    consent = np.random.choice(CONSENT_OPTIONS, p=[0.95, 0.05])
    
    # If no consent, return minimal data
    if consent == 'I do not consent':
        return {
            'Timestamp': timestamp,
            'Participant ID\n(use S0001, S0002, etc)': participant_id,
            'I have read the participant information and consent to take part in this study. I understand I can withdraw at any time': consent,
            'Age bracket': '',
            'Years of professional computer use': '',
            'Prior cybersecurity training?': '',
            'Job role / Department \ne.g., Software Engineer, Finance, Admin': '',
            'On a scale of 1-5, how important is data privacy to you in your professional role?': '',
            'How confident are you in identifying a phishing email?': '',
            'How would you rate your overall digital/computer literacy?': '',
            'I feel stressed at work most days.': '',
            'I tend to follow requests that appear to come from managers.': '',
            'Multitasking frequency': '',
            'Optional comments': '',
            'Would you like to be contacted to withdraw your data if you later request it?': ''
        }
    
    # Demographics
    age = np.random.choice(AGE_BRACKETS, p=AGE_PROBS)
    experience = np.random.choice(YEARS_EXPERIENCE, p=EXPERIENCE_PROBS)
    training = np.random.choice(TRAINING_OPTIONS, p=TRAINING_PROBS)
    job_role = random.choice(JOB_ROLES)
    
    # Determine person's "profile" (affects all ratings)
    # Profile: 0 = very vulnerable, 5 = very secure
    if training == 'Yes':
        base_profile = np.random.normal(3.5, 0.8)  # Trained people are more secure
    else:
        base_profile = np.random.normal(2.5, 0.9)  # Untrained are more vulnerable
    
    # Adjust for age (older = slightly more vulnerable due to lower digital literacy)
    age_index = AGE_BRACKETS.index(age)
    age_adjustment = -0.1 * age_index
    base_profile += age_adjustment
    
    # Adjust for experience (more experience = more secure)
    exp_index = YEARS_EXPERIENCE.index(experience)
    exp_adjustment = 0.2 * exp_index
    base_profile += exp_adjustment
    
    base_profile = max(1, min(5, base_profile))
    
    # Generate correlated scores
    privacy_importance = generate_correlated_scores(base_profile + 0.5, 0.7)  # Most people care about privacy
    phishing_confidence = generate_correlated_scores(base_profile, 0.8)
    digital_literacy = generate_correlated_scores(base_profile, 0.8)
    
    # Stress (inverse correlation - more stressed = more vulnerable)
    stress_level = generate_correlated_scores(5 - base_profile + 0.5, 0.6)
    
    # Authority bias (inverse correlation - higher profile = less bias)
    authority_bias = generate_correlated_scores(5 - base_profile + 1, 0.7)
    
    # Multitasking
    multitask = np.random.choice(MULTITASK_FREQ, p=MULTITASK_PROBS)
    
    # Optional comments (10% chance)
    comments = ''
    if random.random() < 0.10:
        comment_options = [
            '',
            'Interesting survey',
            'Good questions',
            'This is important research',
            'I learned something from this',
            'Thank you',
            'Very relevant to my work',
            'Hope this helps your research'
        ]
        comments = random.choice(comment_options)
    
    # Contact preference (60% say yes)
    contact = random.choice(['', 'Yes, contact me']) if random.random() < 0.6 else ''
    
    return {
        'Timestamp': timestamp,
        'Participant ID\n(use S0001, S0002, etc)': participant_id,
        'I have read the participant information and consent to take part in this study. I understand I can withdraw at any time': consent,
        'Age bracket': age,
        'Years of professional computer use': experience,
        'Prior cybersecurity training?': training,
        'Job role / Department \ne.g., Software Engineer, Finance, Admin': job_role,
        'On a scale of 1-5, how important is data privacy to you in your professional role?': privacy_importance,
        'How confident are you in identifying a phishing email?': phishing_confidence,
        'How would you rate your overall digital/computer literacy?': digital_literacy,
        'I feel stressed at work most days.': stress_level,
        'I tend to follow requests that appear to come from managers.': authority_bias,
        'Multitasking frequency': multitask,
        'Optional comments': comments,
        'Would you like to be contacted to withdraw your data if you later request it?': contact
    }


def add_duplicates_and_quality_issues(df, num_duplicates=8):
    """
    Add realistic data quality issues:
    - Some duplicate responses (same person submitted twice)
    - Some incomplete responses
    """
    # Add duplicates (same participant ID but different timestamps)
    duplicate_indices = random.sample(range(20, len(df)), num_duplicates)
    
    for idx in duplicate_indices:
        original = df.iloc[idx].copy()
        # Same participant ID, slightly different timestamp
        original_time = datetime.strptime(original['Timestamp'], '%d/%m/%Y %H:%M:%S')
        duplicate_time = original_time + timedelta(hours=random.randint(1, 24))
        original['Timestamp'] = duplicate_time.strftime('%d/%m/%Y %H:%M:%S')
        
        # Slightly different responses (people might change their mind)
        if original['How confident are you in identifying a phishing email?'] != '':
            orig_conf = int(original['How confident are you in identifying a phishing email?'])
            original['How confident are you in identifying a phishing email?'] = max(1, min(5, orig_conf + random.randint(-1, 1)))
        
        df = pd.concat([df, pd.DataFrame([original])], ignore_index=True)
    
    return df


def generate_synthetic_dataset(num_responses=400, days_back=20):
    """
    Generate complete synthetic dataset
    """
    print(f"🔄 Generating {num_responses} synthetic survey responses...")
    print(f"📅 Distributed over last {days_back} days")
    
    # Load real data to understand patterns
    real_data = pd.read_csv('Cyber Security Study Participation Form (Responses) - Form responses 1.csv')
    real_patterns = {
        'consent_rate': (real_data.iloc[:, 2] == 'I Consent').mean(),
        'training_rate': (real_data['Prior cybersecurity training?'] == 'Yes').mean() if 'Prior cybersecurity training?' in real_data.columns else 0.6
    }
    
    # Generate responses
    responses = []
    for i in range(num_responses):
        participant_id = f"S{str(i+1).zfill(4)}"
        timestamp = generate_timestamp(days_back, i, num_responses)
        response = generate_realistic_response(participant_id, timestamp, real_patterns)
        responses.append(response)
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Generated {i + 1}/{num_responses} responses...")
    
    # Create DataFrame
    df = pd.DataFrame(responses)
    
    # Add some data quality issues (realistic)
    print("  🔧 Adding realistic data quality issues (duplicates)...")
    df = add_duplicates_and_quality_issues(df, num_duplicates=8)
    
    # Sort by timestamp
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values('Timestamp_dt').drop('Timestamp_dt', axis=1)
    df = df.reset_index(drop=True)
    
    # Generate summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"  Total responses: {len(df)}")
    print(f"  Consented: {(df.iloc[:, 2] == 'I Consent').sum()}")
    print(f"  Non-consented: {(df.iloc[:, 2] == 'I do not consent').sum()}")
    print(f"  With training: {(df['Prior cybersecurity training?'] == 'Yes').sum()}")
    print(f"  Without training: {(df['Prior cybersecurity training?'] == 'No').sum()}")
    print(f"  Date range: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")
    
    # Age distribution
    print(f"\n  Age distribution:")
    for age in AGE_BRACKETS:
        count = (df['Age bracket'] == age).sum()
        print(f"    {age}: {count}")
    
    # Average scores
    consented_df = df[df.iloc[:, 2] == 'I Consent']
    if len(consented_df) > 0:
        print(f"\n  Average scores (consented participants):")
        print(f"    Privacy importance: {consented_df['On a scale of 1-5, how important is data privacy to you in your professional role?'].astype(str).apply(lambda x: float(x) if x.replace('.','',1).isdigit() else np.nan).mean():.2f}")
        print(f"    Phishing confidence: {consented_df['How confident are you in identifying a phishing email?'].astype(str).apply(lambda x: float(x) if x.replace('.','',1).isdigit() else np.nan).mean():.2f}")
        print(f"    Digital literacy: {consented_df['How would you rate your overall digital/computer literacy?'].astype(str).apply(lambda x: float(x) if x.replace('.','',1).isdigit() else np.nan).mean():.2f}")
        print(f"    Stress level: {consented_df['I feel stressed at work most days.'].astype(str).apply(lambda x: float(x) if x.replace('.','',1).isdigit() else np.nan).mean():.2f}")
        print(f"    Authority bias: {consented_df['I tend to follow requests that appear to come from managers.'].astype(str).apply(lambda x: float(x) if x.replace('.','',1).isdigit() else np.nan).mean():.2f}")
    
    return df


def main():
    """Generate and save synthetic dataset"""
    
    print("="*70)
    print("SYNTHETIC SURVEY DATA GENERATOR")
    print("Realistic fake data for ML training and testing")
    print("="*70)
    
    # Generate data
    df = generate_synthetic_dataset(num_responses=NUM_RESPONSES, days_back=DAYS_BACK)
    
    # Save to CSV (backup original first)
    original_file = 'Cyber Security Study Participation Form (Responses) - Form responses 1.csv'
    backup_file = 'Cyber Security Study Participation Form (Responses) - Form responses 1_BACKUP_ORIGINAL.csv'
    synthetic_file = 'Cyber Security Study Participation Form (Responses) - Form responses 1_SYNTHETIC_420.csv'
    
    # Backup original
    import shutil
    print(f"\n💾 Backing up original data to: {backup_file}")
    shutil.copy(original_file, backup_file)
    
    # Save synthetic data
    print(f"💾 Saving synthetic data to: {synthetic_file}")
    df.to_csv(synthetic_file, index=False)
    
    print(f"\n✅ Synthetic dataset generated successfully!")
    print(f"\n📁 Files:")
    print(f"  - Original (backed up): {backup_file}")
    print(f"  - Synthetic (420 responses): {synthetic_file}")
    print(f"  - Original (unchanged): {original_file}")
    
    print(f"\n💡 To use synthetic data with your models:")
    print(f"   1. Copy {synthetic_file} to replace your working data")
    print(f"   2. Or modify data processor to load: {synthetic_file}")
    print(f"   3. Run: python src/data_collection/real_data_processor.py")
    
    print(f"\n⚠️  IMPORTANT: This is SYNTHETIC DATA for testing only!")
    print(f"   For IEEE publication, you MUST use real participant data.")
    print(f"   Synthetic data is useful for:")
    print(f"     - Testing ML models and code")
    print(f"     - Demonstrating the system")
    print(f"     - Debugging and development")
    print(f"     - Estimating required sample sizes")


if __name__ == "__main__":
    main()