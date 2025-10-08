"""
Real Data Processor for Sailaja's Phishing Study
Processes and analyzes actual survey responses collected from participants
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


class RealDataProcessor:
    """
    Process real survey data collected from Google Forms
    """
    
    def __init__(self, data_path: str = 'Cyber Security Study Participation Form (Responses) - Form responses 1.csv'):
        """Initialize with real data file"""
        self.data_path = data_path
        self.data = None
        self.clean_data = None
        
    def load_data(self):
        """Load the CSV file from Google Forms"""
        print("Loading real participant data...")
        self.data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.data)} responses")
        return self.data
    
    def clean_column_names(self):
        """Clean column names for easier processing"""
        column_mapping = {
            'Timestamp': 'timestamp',
            'Participant ID\n(use S0001, S0002, etc)': 'participant_id',
            'I have read the participant information and consent to take part in this study. I understand I can withdraw at any time': 'consent',
            'Age bracket': 'age_bracket',
            'Years of professional computer use': 'years_experience',
            'Prior cybersecurity training?': 'prior_training',
            'Job role / Department \ne.g., Software Engineer, Finance, Admin': 'job_role',
            'On a scale of 1-5, how important is data privacy to you in your professional role?': 'data_privacy_importance',
            'How confident are you in identifying a phishing email?': 'phishing_confidence',
            'How would you rate your overall digital/computer literacy?': 'digital_literacy',
            'I feel stressed at work most days.': 'stress_level',
            'I tend to follow requests that appear to come from managers.': 'authority_bias',
            'Multitasking frequency': 'multitasking',
            'Optional comments': 'comments',
            'Would you like to be contacted to withdraw your data if you later request it?': 'withdrawal_contact'
        }
        
        self.data.rename(columns=column_mapping, inplace=True)
        print("✓ Column names cleaned")
        
    def clean_and_validate_data(self):
        """Clean and validate the data"""
        print("\nCleaning and validating data...")
        
        # Create a copy for cleaning
        self.clean_data = self.data.copy()
        
        # 1. Filter only consented participants
        consented = self.clean_data[self.clean_data['consent'] == 'I Consent'].copy()
        excluded = len(self.clean_data) - len(consented)
        print(f"  - Excluded {excluded} non-consented participant(s)")
        self.clean_data = consented
        
        # 2. Clean participant IDs (handle inconsistent formats)
        self.clean_data['participant_id'] = self.clean_data['participant_id'].str.strip()
        self.clean_data['participant_id'] = self.clean_data['participant_id'].str.replace('S00', 'S0')
        
        # 3. Handle duplicate participant IDs (keep most recent)
        duplicates = self.clean_data[self.clean_data.duplicated(subset=['participant_id'], keep=False)]
        if len(duplicates) > 0:
            print(f"  - Found {len(duplicates)} duplicate participant IDs, keeping most recent responses")
            self.clean_data = self.clean_data.sort_values('timestamp').drop_duplicates(
                subset=['participant_id'], keep='last'
            )
        
        # 4. Convert timestamp to datetime
        self.clean_data['timestamp'] = pd.to_datetime(self.clean_data['timestamp'], format='%d/%m/%Y %H:%M:%S')
        
        # 5. Standardize text fields
        self.clean_data['job_role'] = self.clean_data['job_role'].str.strip()
        self.clean_data['age_bracket'] = self.clean_data['age_bracket'].str.strip()
        
        # 6. Convert numeric fields
        numeric_fields = ['data_privacy_importance', 'phishing_confidence', 
                         'digital_literacy', 'stress_level', 'authority_bias']
        for field in numeric_fields:
            self.clean_data[field] = pd.to_numeric(self.clean_data[field], errors='coerce')
        
        # 7. Map categorical fields to standard values
        self.clean_data['prior_training'] = self.clean_data['prior_training'].map({'Yes': 1, 'No': 0})
        
        # Multitasking frequency mapping
        multitask_map = {
            'Never': 1,
            'Rarely': 2,
            'Sometimes': 3,
            'Often': 4,
            'Always': 5
        }
        self.clean_data['multitasking_score'] = self.clean_data['multitasking'].map(multitask_map)
        
        # 8. Create derived variables
        # Overall vulnerability score (higher = more vulnerable)
        self.clean_data['vulnerability_score'] = (
            (6 - self.clean_data['phishing_confidence']) +  # Lower confidence = higher vulnerability
            (6 - self.clean_data['digital_literacy']) +      # Lower literacy = higher vulnerability
            self.clean_data['stress_level'] +                # Higher stress = higher vulnerability
            self.clean_data['authority_bias'] +              # Higher authority bias = higher vulnerability
            self.clean_data['multitasking_score']            # More multitasking = higher vulnerability
        ) / 5  # Normalize to 1-5 scale
        
        # Risk level categorization
        self.clean_data['risk_level'] = pd.cut(
            self.clean_data['vulnerability_score'],
            bins=[0, 2.5, 3.5, 5],
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"✓ Data cleaned: {len(self.clean_data)} valid participants")
        
        return self.clean_data
    
    def generate_summary_statistics(self):
        """Generate summary statistics of the real data"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS - REAL DATA")
        print("="*60)
        
        print(f"\n📊 Sample Overview:")
        print(f"  Total Participants: {len(self.clean_data)}")
        print(f"  Data Collection Period: {self.clean_data['timestamp'].min().date()} to {self.clean_data['timestamp'].max().date()}")
        
        print(f"\n👥 Demographics:")
        print(f"\nAge Distribution:")
        print(self.clean_data['age_bracket'].value_counts().to_string())
        
        print(f"\n💼 Job Roles:")
        print(self.clean_data['job_role'].value_counts().to_string())
        
        print(f"\n🎓 Training Status:")
        print(f"  Has Prior Training: {self.clean_data['prior_training'].sum()} ({self.clean_data['prior_training'].mean()*100:.1f}%)")
        print(f"  No Prior Training: {(1-self.clean_data['prior_training']).sum()} ({(1-self.clean_data['prior_training'].mean())*100:.1f}%)")
        
        print(f"\n📈 Key Metrics (Mean ± SD):")
        metrics = {
            'Data Privacy Importance': 'data_privacy_importance',
            'Phishing Confidence': 'phishing_confidence',
            'Digital Literacy': 'digital_literacy',
            'Stress Level': 'stress_level',
            'Authority Bias': 'authority_bias',
            'Vulnerability Score': 'vulnerability_score'
        }
        
        for label, col in metrics.items():
            mean = self.clean_data[col].mean()
            std = self.clean_data[col].std()
            print(f"  {label}: {mean:.2f} ± {std:.2f}")
        
        print(f"\n⚠️ Risk Level Distribution:")
        print(self.clean_data['risk_level'].value_counts().to_string())
        
        print(f"\n🔄 Multitasking Frequency:")
        print(self.clean_data['multitasking'].value_counts().to_string())
        
    def identify_high_risk_participants(self):
        """Identify participants at high risk"""
        print("\n" + "="*60)
        print("HIGH RISK PARTICIPANTS")
        print("="*60)
        
        high_risk = self.clean_data[self.clean_data['risk_level'] == 'High'].copy()
        
        if len(high_risk) > 0:
            print(f"\n⚠️ Found {len(high_risk)} HIGH RISK participants:")
            print("\nParticipant Details:")
            display_cols = ['participant_id', 'job_role', 'phishing_confidence', 
                          'digital_literacy', 'stress_level', 'vulnerability_score', 'risk_level']
            print(high_risk[display_cols].to_string(index=False))
        else:
            print("\n✓ No high-risk participants identified")
        
        return high_risk
    
    def analyze_by_training_status(self):
        """Compare metrics by training status"""
        print("\n" + "="*60)
        print("ANALYSIS BY TRAINING STATUS")
        print("="*60)
        
        comparison_metrics = ['phishing_confidence', 'digital_literacy', 'vulnerability_score']
        
        for metric in comparison_metrics:
            trained = self.clean_data[self.clean_data['prior_training'] == 1][metric]
            untrained = self.clean_data[self.clean_data['prior_training'] == 0][metric]
            
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  With Training: {trained.mean():.2f} ± {trained.std():.2f} (n={len(trained)})")
            print(f"  No Training: {untrained.mean():.2f} ± {untrained.std():.2f} (n={len(untrained)})")
            
            if len(trained) > 0 and len(untrained) > 0:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(trained.dropna(), untrained.dropna())
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"  T-test: t={t_stat:.2f}, p={p_value:.4f} {significance}")
    
    def save_processed_data(self, output_dir: str = 'data/processed'):
        """Save cleaned and processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned data
        output_path = f'{output_dir}/real_survey_data_cleaned.csv'
        self.clean_data.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned data saved to: {output_path}")
        
        # Save summary report
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_participants': len(self.clean_data),
            'date_range': {
                'start': self.clean_data['timestamp'].min().isoformat(),
                'end': self.clean_data['timestamp'].max().isoformat()
            },
            'demographics': {
                'age_distribution': self.clean_data['age_bracket'].value_counts().to_dict(),
                'training_status': {
                    'with_training': int(self.clean_data['prior_training'].sum()),
                    'without_training': int((1-self.clean_data['prior_training']).sum())
                }
            },
            'risk_distribution': self.clean_data['risk_level'].value_counts().to_dict(),
            'mean_scores': {
                'vulnerability_score': float(self.clean_data['vulnerability_score'].mean()),
                'phishing_confidence': float(self.clean_data['phishing_confidence'].mean()),
                'digital_literacy': float(self.clean_data['digital_literacy'].mean()),
                'stress_level': float(self.clean_data['stress_level'].mean())
            }
        }
        
        report_path = f'{output_dir}/real_data_summary_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Summary report saved to: {report_path}")
        
        return output_path, report_path


def main():
    """Main processing pipeline for real data"""
    print("="*60)
    print("REAL DATA PROCESSOR - PHISHING STUDY")
    print("Sailaja Midde - MSc Cybersecurity Research")
    print("="*60)
    
    # Initialize processor
    processor = RealDataProcessor()
    
    # Load data
    processor.load_data()
    
    # Clean column names
    processor.clean_column_names()
    
    # Clean and validate
    processor.clean_and_validate_data()
    
    # Generate statistics
    processor.generate_summary_statistics()
    
    # Identify high-risk participants
    processor.identify_high_risk_participants()
    
    # Analyze by training
    processor.analyze_by_training_status()
    
    # Save processed data
    processor.save_processed_data()
    
    print("\n" + "="*60)
    print("✓ DATA PROCESSING COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review processed data in: data/processed/")
    print("2. Run phishing simulations with participants")
    print("3. Merge survey data with experiment results")
    print("4. Run statistical analysis and ML models")
    print("5. Launch dashboard to visualize results")


if __name__ == "__main__":
    main()
