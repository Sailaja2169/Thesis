"""
Enhanced Survey Generator with ALL CA1 Requirements
Includes urgency bias, familiarity bias, training details, workload, and education
"""

import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, List
import json


class EnhancedSurveyGenerator:
    """
    Complete survey covering ALL items from CA1 document
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize enhanced survey generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.survey_questions = self._generate_complete_questions()
    
    def _generate_complete_questions(self) -> Dict:
        """Generate COMPLETE survey questions addressing all CA1 items"""
        
        questions = {
            # ITEM #4: Demographics and Professional Factors
            "demographics": [
                {
                    "id": "D1",
                    "question": "What is your age bracket?",
                    "type": "single_choice",
                    "options": ["18–25", "26-35", "36-45", "46-55", "56+"],
                    "required": True,
                    "ca1_item": "Item #4: Demographics"
                },
                {
                    "id": "D2",
                    "question": "What is your highest level of education?",
                    "type": "single_choice",
                    "options": ["High School", "Associate/Diploma", "Bachelor's Degree", 
                               "Master's Degree", "PhD/Doctorate", "Professional Certification"],
                    "required": True,
                    "ca1_item": "Item #4: Demographics"
                },
                {
                    "id": "D3",
                    "question": "What is your current job role or department?",
                    "type": "text",
                    "required": True,
                    "ca1_item": "Item #4: Professional Factors"
                },
                {
                    "id": "D4",
                    "question": "How many years of professional computer use do you have?",
                    "type": "single_choice",
                    "options": ["0–1", "1-3", "3-5", "5+"],
                    "required": True,
                    "ca1_item": "Item #4: Professional Experience"
                },
                {
                    "id": "D5",
                    "question": "How would you rate your overall digital/computer literacy?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Very Low)", "2 (Low)", "3 (Moderate)", "4 (High)", "5 (Very High)"],
                    "required": True,
                    "ca1_item": "Item #4: Digital Literacy"
                }
            ],
            
            # ITEM #1: Cognitive Biases and Decisions
            "cognitive_biases": [
                {
                    "id": "CB1",
                    "question": "I tend to follow requests that appear to come from managers or authority figures.",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Strongly Disagree)", "2 (Disagree)", "3 (Neutral)", 
                              "4 (Agree)", "5 (Strongly Agree)"],
                    "required": True,
                    "ca1_item": "Item #1: Authority Bias",
                    "research_question": "RQ1"
                },
                {
                    "id": "CB2",
                    "question": "How likely are you to act immediately on urgent or time-sensitive emails (e.g., 'Your account will be suspended', 'Immediate action required')?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Very Unlikely)", "2 (Unlikely)", "3 (Neutral)", 
                              "4 (Likely)", "5 (Very Likely)"],
                    "required": True,
                    "ca1_item": "Item #1: Urgency Effect",
                    "research_question": "RQ1"
                },
                {
                    "id": "CB3",
                    "question": "Do you tend to trust and open emails from familiar-looking senders without verifying the actual email address?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Never)", "2 (Rarely)", "3 (Sometimes)", 
                              "4 (Often)", "5 (Always)"],
                    "required": True,
                    "ca1_item": "Item #1: Familiarity Bias",
                    "research_question": "RQ1"
                },
                {
                    "id": "CB4",
                    "question": "How often do you click on intriguing or surprising email content (e.g., 'You've won a prize', 'Unexpected package delivery')?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Never)", "2 (Rarely)", "3 (Sometimes)", 
                              "4 (Often)", "5 (Always)"],
                    "required": True,
                    "ca1_item": "Item #1: Curiosity Bias",
                    "research_question": "RQ1"
                }
            ],
            
            # ITEM #2: Psychological State and Environment
            "work_context": [
                {
                    "id": "WC1",
                    "question": "I feel stressed at work most days.",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Strongly Disagree)", "2 (Disagree)", "3 (Neutral)", 
                              "4 (Agree)", "5 (Strongly Agree)"],
                    "required": True,
                    "ca1_item": "Item #2: Stress Level",
                    "research_question": "RQ2"
                },
                {
                    "id": "WC2",
                    "question": "How often do you multitask while checking your work emails?",
                    "type": "single_choice",
                    "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True,
                    "ca1_item": "Item #2: Multitasking",
                    "research_question": "RQ2"
                },
                {
                    "id": "WC3",
                    "question": "On average, how many work-related emails do you receive per day?",
                    "type": "single_choice",
                    "options": ["0-20", "21-50", "51-100", "101-200", "200+"],
                    "required": True,
                    "ca1_item": "Item #2: Workload",
                    "research_question": "RQ2"
                },
                {
                    "id": "WC4",
                    "question": "How would you describe your typical workload?",
                    "type": "single_choice",
                    "options": ["Very Light", "Light", "Moderate", "Heavy", "Very Heavy"],
                    "required": True,
                    "ca1_item": "Item #2: Workload Perception",
                    "research_question": "RQ2"
                },
                {
                    "id": "WC5",
                    "question": "Do you often work under tight deadlines?",
                    "type": "single_choice",
                    "options": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True,
                    "ca1_item": "Item #2: Time Pressure",
                    "research_question": "RQ2"
                }
            ],
            
            # ITEM #3: Training and Awareness
            "training": [
                {
                    "id": "T1",
                    "question": "Have you received any cybersecurity or phishing awareness training?",
                    "type": "single_choice",
                    "options": ["Yes", "No"],
                    "required": True,
                    "ca1_item": "Item #3: Training Status",
                    "research_question": "RQ3"
                },
                {
                    "id": "T2",
                    "question": "If yes, when did you last receive cybersecurity training?",
                    "type": "single_choice",
                    "options": ["Within last 3 months", "3-6 months ago", "6-12 months ago", 
                               "1-2 years ago", "More than 2 years ago", "N/A - No training"],
                    "required": False,
                    "ca1_item": "Item #3: Training Recency",
                    "research_question": "RQ3"
                },
                {
                    "id": "T3",
                    "question": "What type of cybersecurity training did you receive? (Select all that apply)",
                    "type": "multiple_choice",
                    "options": ["Online course/module", "In-person workshop", "Company-mandated training", 
                               "Self-study", "Phishing simulation exercises", "Certification program", 
                               "University course", "None"],
                    "required": False,
                    "ca1_item": "Item #3: Training Type",
                    "research_question": "RQ3"
                },
                {
                    "id": "T4",
                    "question": "Approximately how long was your most recent training?",
                    "type": "single_choice",
                    "options": ["Less than 30 minutes", "30-60 minutes", "1-2 hours", 
                               "2-4 hours", "More than 4 hours", "N/A"],
                    "required": False,
                    "ca1_item": "Item #3: Training Duration",
                    "research_question": "RQ3"
                },
                {
                    "id": "T5",
                    "question": "How confident are you in identifying a phishing email?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Not Confident)", "2 (Slightly Confident)", "3 (Moderately Confident)", 
                              "4 (Very Confident)", "5 (Extremely Confident)"],
                    "required": True,
                    "ca1_item": "Item #3: Self-assessed Competence",
                    "research_question": "RQ3"
                },
                {
                    "id": "T6",
                    "question": "How would you rate the effectiveness of your cybersecurity training (if received)?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Not Effective)", "2", "3 (Moderately Effective)", "4", "5 (Very Effective)", "N/A"],
                    "required": False,
                    "ca1_item": "Item #3: Training Effectiveness",
                    "research_question": "RQ3"
                }
            ],
            
            # Phishing Experience and Behavior
            "phishing_experience": [
                {
                    "id": "PE1",
                    "question": "Have you ever fallen victim to a phishing attack?",
                    "type": "single_choice",
                    "options": ["Yes", "No", "Not sure"],
                    "required": True,
                    "ca1_item": "Background Context"
                },
                {
                    "id": "PE2",
                    "question": "Have you ever reported a suspected phishing email to IT/security?",
                    "type": "single_choice",
                    "options": ["Yes, frequently", "Yes, occasionally", "Yes, once", "No, never"],
                    "required": True,
                    "ca1_item": "Behavioral Response"
                },
                {
                    "id": "PE3",
                    "question": "How often do you encounter suspicious or potential phishing emails?",
                    "type": "single_choice",
                    "options": ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
                    "required": True,
                    "ca1_item": "Exposure Frequency"
                }
            ],
            
            # Security Behavior Indicators
            "security_behavior": [
                {
                    "id": "SB1",
                    "question": "Before clicking a link in an email, I verify the actual URL by hovering over it.",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Never)", "2 (Rarely)", "3 (Sometimes)", 
                              "4 (Often)", "5 (Always)"],
                    "required": True,
                    "ca1_item": "Protective Behavior"
                },
                {
                    "id": "SB2",
                    "question": "I carefully check the sender's email address before responding to emails.",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Never)", "2 (Rarely)", "3 (Sometimes)", 
                              "4 (Often)", "5 (Always)"],
                    "required": True,
                    "ca1_item": "Protective Behavior"
                },
                {
                    "id": "SB3",
                    "question": "I check for spelling and grammar errors in emails as potential phishing indicators.",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Never)", "2 (Rarely)", "3 (Sometimes)", 
                              "4 (Often)", "5 (Always)"],
                    "required": True,
                    "ca1_item": "Protective Behavior"
                },
                {
                    "id": "SB4",
                    "question": "On a scale of 1-5, how important is data privacy to you in your professional role?",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "labels": ["1 (Not Important)", "2", "3 (Moderately Important)", "4", "5 (Very Important)"],
                    "required": True,
                    "ca1_item": "Privacy Awareness"
                }
            ]
        }
        
        return questions
    
    def export_to_google_forms_csv(self, output_path: str = 'data/enhanced_survey_google_forms.csv'):
        """Export enhanced survey in Google Forms import format"""
        rows = []
        
        for section_name, questions in self.survey_questions.items():
            for q in questions:
                row = {
                    'Section': section_name.replace('_', ' ').title(),
                    'Question ID': q['id'],
                    'Question Text': q['question'],
                    'Question Type': q['type'],
                    'Required': 'Yes' if q.get('required', False) else 'No',
                    'Options': ' | '.join(q.get('options', [])) if 'options' in q else '',
                    'CA1 Item': q.get('ca1_item', ''),
                    'Research Question': q.get('research_question', ''),
                    'Scale Labels': ' | '.join(q.get('labels', [])) if 'labels' in q else ''
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✓ Enhanced survey exported to: {output_path}")
        print(f"✓ Total questions: {len(df)}")
        print(f"\n📊 Questions by CA1 Item:")
        print(df['CA1 Item'].value_counts().to_string())
        print(f"\n📊 Questions by Research Question:")
        print(df['Research Question'].value_counts().to_string())
        
        return df
    
    def generate_comparison_report(self):
        """Generate report comparing old vs new survey"""
        total_questions = sum(len(questions) for questions in self.survey_questions.values())
        
        print("\n" + "="*60)
        print("ENHANCED SURVEY COMPARISON REPORT")
        print("="*60)
        
        print(f"\n📝 Survey Coverage:")
        print(f"  Total Questions: {total_questions}")
        print(f"  Demographics: {len(self.survey_questions['demographics'])}")
        print(f"  Cognitive Biases: {len(self.survey_questions['cognitive_biases'])}")
        print(f"  Work Context: {len(self.survey_questions['work_context'])}")
        print(f"  Training & Awareness: {len(self.survey_questions['training'])}")
        print(f"  Phishing Experience: {len(self.survey_questions['phishing_experience'])}")
        print(f"  Security Behavior: {len(self.survey_questions['security_behavior'])}")
        
        print(f"\n✅ NEW QUESTIONS ADDED:")
        print("  1. ✅ Education Level (D2) - Item #4")
        print("  2. ✅ Urgency Bias (CB2) - Item #1, RQ1")
        print("  3. ✅ Familiarity Bias (CB3) - Item #1, RQ1")
        print("  4. ✅ Curiosity Bias (CB4) - Item #1, RQ1")
        print("  5. ✅ Email Volume (WC3) - Item #2, RQ2")
        print("  6. ✅ Workload Perception (WC4) - Item #2, RQ2")
        print("  7. ✅ Time Pressure/Deadlines (WC5) - Item #2, RQ2")
        print("  8. ✅ Training Recency (T2) - Item #3, RQ3")
        print("  9. ✅ Training Type (T3) - Item #3, RQ3")
        print("  10. ✅ Training Duration (T4) - Item #3, RQ3")
        print("  11. ✅ Training Effectiveness (T6) - Item #3, RQ3")
        
        print(f"\n📊 CA1 Requirements Coverage:")
        print("  ✅ Item #1 (Cognitive Biases): 4/4 biases covered")
        print("  ✅ Item #2 (Psychological State): All factors covered")
        print("  ✅ Item #3 (Training): Comprehensive coverage")
        print("  ✅ Item #4 (Demographics): Complete coverage")
        
        print(f"\n🎯 Research Questions Coverage:")
        print("  ✅ RQ1 (Cognitive Biases): 4 questions")
        print("  ✅ RQ2 (Stress/Multitasking): 5 questions")
        print("  ✅ RQ3 (Training): 6 questions")
        print("  ✅ RQ4 (Demographics): 5 questions")


def main():
    """Generate enhanced survey with all CA1 requirements"""
    print("="*60)
    print("ENHANCED SURVEY GENERATOR")
    print("Complete CA1 Requirement Coverage")
    print("="*60)
    
    generator = EnhancedSurveyGenerator()
    
    # Export to Google Forms format
    print("\n1. Exporting enhanced survey...")
    generator.export_to_google_forms_csv()
    
    # Generate comparison report
    print("\n2. Generating comparison report...")
    generator.generate_comparison_report()
    
    print("\n" + "="*60)
    print("✓ ENHANCED SURVEY GENERATION COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Review 'data/enhanced_survey_google_forms.csv'")
    print("2. Import to Google Forms")
    print("3. Deploy to NEW participants OR")
    print("4. Send as follow-up to existing participants")
    print("5. Merge with existing data for complete analysis")


if __name__ == "__main__":
    main()