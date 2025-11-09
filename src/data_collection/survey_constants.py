"""
This file contains constants for survey question column names.
Using constants helps prevent typos and makes the code easier to maintain.
"""

# --- Metadata ---
TIMESTAMP = "Timestamp"
EMAIL_ADDRESS = "Email Address"
CONSENT = "I consent to my data being used for this research study."

# --- Cognitive Biases (CB) ---
CB_AUTHORITY = "CB1: Authority Bias"
CB_URGENCY = "CB2: Urgency Bias"
CB_FAMILIARITY = "CB3: Familiarity Bias"
CB_CURIOSITY = "CB4: Curiosity Bias"

# --- Work Context (WC) ---
WC_STRESS = "WC1: Stress Level"
WC_MULTITASKING = "WC2: Multitasking Frequency"
WC_EMAIL_VOLUME = "WC3: Email Volume"
WC_WORKLOAD = "WC4: Workload Perception"
WC_TIME_PRESSURE = "WC5: Time Pressure"

# --- Training (T) ---
T_PRIOR_TRAINING = "T1: Prior cybersecurity training?"
T_RECENCY = "T2: Training Recency"
T_TYPE = "T3: Training Type"
T_DURATION = "T4: Training Duration"
T_CONFIDENCE = "T5: Phishing Confidence"
T_EFFECTIVENESS = "T6: Training Effectiveness"

# --- Demographics (D) ---
D_AGE = "D1: Age"
D_EDUCATION = "D2: Education Level"
D_JOB_ROLE = "D3: Job Role"
D_EXPERIENCE = "D4: Years of Work Experience"
D_DIGITAL_LITERACY = "D5: Digital Literacy"

# --- Security Habits (H) ---
H_SECURITY_SOFTWARE = "H1: Security Software Usage"
H_PASSWORD_MANAGER = "H2: Password Manager Usage"
H_UPDATE_FREQUENCY = "H3: Software Update Frequency"
H_BACKUP_FREQUENCY = "H4: Data Backup Frequency"

# --- Phishing Experience (P) ---
P_SUSPICIOUS_EMAILS = "P1: Frequency of Suspicious Emails"
P_PHISHING_VICTIM = "P2: Previous Phishing Victim"
P_REPORTING = "P3: Reporting Suspicious Emails"

# --- Behaviors (B) ---
B_CLICK_LINK = "B1: Click Behavior"
B_REPORT_EMAIL = "B2: Report Behavior"
B_SHARE_INFO = "B3: Information Sharing Behavior"

# --- Column Lists ---
ALL_COLUMNS = [
    TIMESTAMP, EMAIL_ADDRESS, CONSENT,
    CB_AUTHORITY, CB_URGENCY, CB_FAMILIARITY, CB_CURIOSITY,
    WC_STRESS, WC_MULTITASKING, WC_EMAIL_VOLUME, WC_WORKLOAD, WC_TIME_PRESSURE,
    T_PRIOR_TRAINING, T_RECENCY, T_TYPE, T_DURATION, T_CONFIDENCE, T_EFFECTIVENESS,
    D_AGE, D_EDUCATION, D_JOB_ROLE, D_EXPERIENCE, D_DIGITAL_LITERACY,
    H_SECURITY_SOFTWARE, H_PASSWORD_MANAGER, H_UPDATE_FREQUENCY, H_BACKUP_FREQUENCY,
    P_SUSPICIOUS_EMAILS, P_PHISHING_VICTIM, P_REPORTING,
    B_CLICK_LINK, B_REPORT_EMAIL, B_SHARE_INFO
]