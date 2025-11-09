"""
Data Collection Module - Survey Generator
Generates comprehensive surveys for phishing susceptibility research
"""

import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, List
import json


class SurveyGenerator:
    """
    Generates structured surveys for phishing behavioral research
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize survey generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.survey_questions = self._generate_questions()
    
    def _generate_questions(self) -> Dict:
        """Generate comprehensive survey questions"""
        
        questions = {
            # Demographics
            "demographics": [
                {
                    "id": "D1",
                    "question": "What is your age group?",
                    "type": "single_choice",
                    "options": self.config['demographics']['age_groups'],
                    "required": True
                },
                {
                    "id": "D2",
                    "question": "What is your current job role?",
                    "type": "single_choice",
                    "options": self.config['demographics']['job_roles'],
                    "required": True
                },
                {
                    "id": "D3",
                    "question": "What is your highest level of education?",
                    "type": "single_choice",
                    "options": self.config['demographics']['education_levels'],
                    "required": True
                },
                {
                    "id": "D4",
                    "question": "How many years of work experience do you have?",
                    "type": "number",
                    "required": True
                },
                {
                    "id": "D5",
                    "question": "Rate your digital literacy (1-10)",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "required": True
                }
            ],
            
            # Cognitive Biases
            "cognitive_biases": [
                {
                    "id": "CB1",
                    "question": "How likely are you to respond to emails from authority figures (e.g., CEO, manager)?",
                    "type": "likert",
                    "scale": ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"],
                    "required": True
                },
                {
                    "id": "CB2",
                    "question": "How do you typically react to urgent emails requiring immediate action?",
                    "type": "likert",
                    "scale": ["Verify first", "Sometimes verify", "Neutral", "Usually act quickly", "Always act immediately"],
                    "required": True
                },
                {
                    "id": "CB3",
                    "question": "Do you tend to click on links from familiar-looking senders without verification?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                },
                {
                    "id": "CB4",
                    "question": "How often do you open emails with intriguing subject lines (e.g., 'You've won a prize')?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                }
            ],
            
            # Work Context and Stress
            "work_context": [
                {
                    "id": "WC1",
                    "question": "On a typical workday, how stressed do you feel?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "required": True
                },
                {
                    "id": "WC2",
                    "question": "How often do you multitask while checking emails?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                },
                {
                    "id": "WC3",
                    "question": "How many emails do you receive per day on average?",
                    "type": "single_choice",
                    "options": ["0-20", "21-50", "51-100", "101-200", "200+"],
                    "required": True
                },
                {
                    "id": "WC4",
                    "question": "Do you check work emails outside of work hours?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                }
            ],
            
            # Cybersecurity Training
            "training": [
                {
                    "id": "T1",
                    "question": "Have you received cybersecurity training in the past 12 months?",
                    "type": "yes_no",
                    "required": True
                },
                {
                    "id": "T2",
                    "question": "If yes, how many training sessions have you attended?",
                    "type": "number",
                    "required": False
                },
                {
                    "id": "T3",
                    "question": "How would you rate the effectiveness of your cybersecurity training?",
                    "type": "scale",
                    "min": 1,
                    "max": 10,
                    "required": False
                },
                {
                    "id": "T4",
                    "question": "How confident are you in identifying phishing emails?",
                    "type": "likert",
                    "scale": ["Not at all confident", "Slightly confident", "Moderately confident", "Very confident", "Extremely confident"],
                    "required": True
                }
            ],
            
            # Phishing Experience
            "experience": [
                {
                    "id": "E1",
                    "question": "Have you ever fallen victim to a phishing attack?",
                    "type": "yes_no",
                    "required": True
                },
                {
                    "id": "E2",
                    "question": "Have you ever reported a suspected phishing email?",
                    "type": "yes_no",
                    "required": True
                },
                {
                    "id": "E3",
                    "question": "How often do you encounter suspicious emails?",
                    "type": "single_choice",
                    "options": ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
                    "required": True
                }
            ],
            
            # Behavioral Indicators
            "behavioral": [
                {
                    "id": "B1",
                    "question": "Do you verify sender email addresses before responding?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                },
                {
                    "id": "B2",
                    "question": "Do you hover over links to check URLs before clicking?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                },
                {
                    "id": "B3",
                    "question": "Do you check for spelling/grammar errors in emails?",
                    "type": "likert",
                    "scale": ["Never", "Rarely", "Sometimes", "Often", "Always"],
                    "required": True
                }
            ]
        }
        
        return questions
    
    def export_to_json(self, output_path: str = 'data/survey_template.json'):
        """Export survey to JSON format"""
        survey_data = {
            "title": "Phishing Susceptibility Survey",
            "description": "Research survey on human factors in phishing attacks",
            "consent": "By completing this survey, you consent to participate in this research study...",
            "estimated_time": "15-20 minutes",
            "sections": self.survey_questions,
            "generated": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(survey_data, f, indent=2)
        
        print(f"Survey template exported to {output_path}")
        return survey_data
    
    def export_to_google_forms_format(self, output_path: str = 'data/google_forms_import.csv'):
        """
        Export in a format suitable for Google Forms import
        """
        rows = []
        
        for section_name, questions in self.survey_questions.items():
            for q in questions:
                row = {
                    'Section': section_name.replace('_', ' ').title(),
                    'Question ID': q['id'],
                    'Question': q['question'],
                    'Type': q['type'],
                    'Required': 'Yes' if q.get('required', False) else 'No',
                    'Options': '|'.join(q.get('options', [])) if 'options' in q else ''
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Google Forms format exported to {output_path}")
        return df
    
    def generate_sample_response(self) -> Dict:
        """Generate a sample survey response for testing"""
        import random
        
        response = {
            "response_id": f"RESP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "answers": {}
        }
        
        for section_name, questions in self.survey_questions.items():
            for q in questions:
                q_id = q['id']
                q_type = q['type']
                
                if q_type == 'single_choice':
                    response['answers'][q_id] = random.choice(q['options'])
                elif q_type == 'yes_no':
                    response['answers'][q_id] = random.choice(['Yes', 'No'])
                elif q_type == 'scale':
                    response['answers'][q_id] = random.randint(q['min'], q['max'])
                elif q_type == 'number':
                    response['answers'][q_id] = random.randint(0, 15)
                elif q_type == 'likert':
                    response['answers'][q_id] = random.choice(q['scale'])
        
        return response
    
    def generate_sample_dataset(self, n_responses: int = 100, 
                               output_path: str = 'data/synthetic/sample_responses.csv'):
        """Generate a synthetic dataset for testing"""
        responses = []
        
        for i in range(n_responses):
            response = self.generate_sample_response()
            flat_response = {'response_id': response['response_id'], 
                           'timestamp': response['timestamp']}
            flat_response.update(response['answers'])
            responses.append(flat_response)
        
        df = pd.DataFrame(responses)
        df.to_csv(output_path, index=False)
        print(f"Generated {n_responses} sample responses and saved to {output_path}")
        return df


def main():
    """Main function to generate survey templates"""
    print("Initializing Survey Generator...")
    
    # Create survey generator
    generator = SurveyGenerator()
    
    # Export to different formats
    print("\n1. Exporting to JSON format...")
    generator.export_to_json()
    
    print("\n2. Exporting to Google Forms format...")
    generator.export_to_google_forms_format()
    
    print("\n3. Generating sample dataset...")
    generator.generate_sample_dataset(n_responses=300)
    
    print("\n✓ Survey generation complete!")
    print("\nNext steps:")
    print("1. Review survey_template.json")
    print("2. Import google_forms_import.csv to Google Forms")
    print("3. Use sample_responses.csv for testing analysis pipelines")


if __name__ == "__main__":
    main()