"""
Phishing Scenario Experiment Simulator
Simulates phishing attacks in controlled environment for research purposes
"""

import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random


class PhishingScenarioSimulator:
    """
    Simulates phishing scenarios with varying difficulty levels and cognitive triggers
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize simulator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> List[Dict]:
        """Create diverse phishing scenarios"""
        
        scenarios = [
            # Authority Bias Scenarios
            {
                "id": "PS001",
                "difficulty": "easy",
                "cognitive_trigger": "authority_bias",
                "subject": "URGENT: CEO Requests Immediate Action",
                "sender": "ceo@company-urgent.com",
                "content": "This is John Smith, CEO. I need you to urgently process a payment...",
                "phishing_indicators": [
                    "suspicious_domain",
                    "urgency_language",
                    "unusual_request"
                ],
                "legitimate_indicators": [],
                "correct_response": "report_phishing"
            },
            {
                "id": "PS002",
                "difficulty": "medium",
                "cognitive_trigger": "authority_bias",
                "subject": "IT Security Update Required",
                "sender": "it.support@company-secure.com",
                "content": "Our IT department requires all staff to update security credentials...",
                "phishing_indicators": [
                    "suspicious_link",
                    "credential_request"
                ],
                "legitimate_indicators": [
                    "professional_formatting"
                ],
                "correct_response": "report_phishing"
            },
            {
                "id": "PS003",
                "difficulty": "hard",
                "cognitive_trigger": "authority_bias",
                "subject": "Re: Q4 Budget Review",
                "sender": "j.smith@company.com",
                "content": "Following up on our meeting. Please review the attached budget spreadsheet...",
                "phishing_indicators": [
                    "spoofed_internal_email",
                    "malicious_attachment"
                ],
                "legitimate_indicators": [
                    "correct_email_format",
                    "professional_tone",
                    "contextual_reference"
                ],
                "correct_response": "report_phishing"
            },
            
            # Urgency Bias Scenarios
            {
                "id": "PS004",
                "difficulty": "easy",
                "cognitive_trigger": "urgency_bias",
                "subject": "Your account will be closed in 24 hours!",
                "sender": "noreply@bank-security.com",
                "content": "URGENT ACTION REQUIRED! Your account has been flagged...",
                "phishing_indicators": [
                    "extreme_urgency",
                    "threat_language",
                    "suspicious_domain"
                ],
                "legitimate_indicators": [],
                "correct_response": "report_phishing"
            },
            {
                "id": "PS005",
                "difficulty": "medium",
                "cognitive_trigger": "urgency_bias",
                "subject": "Expiring: Your package is waiting",
                "sender": "delivery@shipping-service.com",
                "content": "Your package requires action within 48 hours or will be returned...",
                "phishing_indicators": [
                    "urgency_language",
                    "suspicious_tracking_link"
                ],
                "legitimate_indicators": [
                    "tracking_number"
                ],
                "correct_response": "report_phishing"
            },
            
            # Curiosity/Reward Scenarios
            {
                "id": "PS006",
                "difficulty": "easy",
                "cognitive_trigger": "curiosity_bias",
                "subject": "You've won $1,000,000!",
                "sender": "winner@lottery-claim.com",
                "content": "Congratulations! You've been selected as our grand prize winner...",
                "phishing_indicators": [
                    "too_good_to_be_true",
                    "suspicious_domain",
                    "request_personal_info"
                ],
                "legitimate_indicators": [],
                "correct_response": "report_phishing"
            },
            {
                "id": "PS007",
                "difficulty": "medium",
                "cognitive_trigger": "curiosity_bias",
                "subject": "Your Amazon order confirmation #18293847",
                "sender": "orders@amazon-services.com",
                "content": "Thank you for your order of iPhone 15 Pro Max ($1,299.99)...",
                "phishing_indicators": [
                    "suspicious_domain",
                    "unexpected_purchase"
                ],
                "legitimate_indicators": [
                    "order_number",
                    "professional_formatting"
                ],
                "correct_response": "report_phishing"
            },
            
            # Familiarity/Social Engineering
            {
                "id": "PS008",
                "difficulty": "hard",
                "cognitive_trigger": "familiarity_bias",
                "subject": "Re: Meeting notes from yesterday",
                "sender": "colleague@company.com",
                "content": "Hi, here are the meeting notes we discussed. Please review the document...",
                "phishing_indicators": [
                    "spoofed_colleague_email",
                    "suspicious_attachment"
                ],
                "legitimate_indicators": [
                    "internal_email_format",
                    "casual_tone",
                    "contextual_reference"
                ],
                "correct_response": "verify_with_sender"
            },
            
            # Legitimate Emails (Control)
            {
                "id": "PS009",
                "difficulty": "medium",
                "cognitive_trigger": "none",
                "subject": "Team meeting scheduled for tomorrow 2 PM",
                "sender": "calendar@company.com",
                "content": "You have been invited to: Q4 Planning Meeting...",
                "phishing_indicators": [],
                "legitimate_indicators": [
                    "internal_sender",
                    "calendar_invitation",
                    "professional_formatting"
                ],
                "correct_response": "safe_to_open"
            },
            {
                "id": "PS010",
                "difficulty": "easy",
                "cognitive_trigger": "none",
                "subject": "Your monthly report is ready",
                "sender": "reports@company.com",
                "content": "Your automated monthly performance report has been generated...",
                "phishing_indicators": [],
                "legitimate_indicators": [
                    "internal_system",
                    "expected_email",
                    "secure_link"
                ],
                "correct_response": "safe_to_open"
            }
        ]
        
        return scenarios
    
    def run_experiment(self, participant_id: str, scenario_ids: List[str] = None) -> Dict:
        """
        Run phishing experiment for a participant
        
        Args:
            participant_id: Unique identifier for participant
            scenario_ids: List of scenario IDs to test (None = all scenarios)
        
        Returns:
            Experiment results dictionary
        """
        if scenario_ids is None:
            # Randomly select scenarios with balanced difficulty
            scenarios_to_test = self._select_balanced_scenarios()
        else:
            scenarios_to_test = [s for s in self.scenarios if s['id'] in scenario_ids]
        
        experiment_results = {
            "participant_id": participant_id,
            "start_time": datetime.now().isoformat(),
            "scenarios": [],
            "performance_metrics": {}
        }
        
        for scenario in scenarios_to_test:
            result = self._present_scenario(scenario)
            experiment_results["scenarios"].append(result)
        
        # Calculate metrics
        experiment_results["end_time"] = datetime.now().isoformat()
        experiment_results["performance_metrics"] = self._calculate_metrics(
            experiment_results["scenarios"]
        )
        
        return experiment_results
    
    def _select_balanced_scenarios(self, n_scenarios: int = 10) -> List[Dict]:
        """Select balanced set of scenarios across difficulty levels"""
        easy = [s for s in self.scenarios if s['difficulty'] == 'easy']
        medium = [s for s in self.scenarios if s['difficulty'] == 'medium']
        hard = [s for s in self.scenarios if s['difficulty'] == 'hard']
        
        selected = []
        selected.extend(random.sample(easy, min(3, len(easy))))
        selected.extend(random.sample(medium, min(4, len(medium))))
        selected.extend(random.sample(hard, min(3, len(hard))))
        
        random.shuffle(selected)
        return selected[:n_scenarios]
    
    def _present_scenario(self, scenario: Dict) -> Dict:
        """
        Simulate presenting a scenario and recording response
        In real implementation, this would display actual emails
        """
        # Simulate participant response time and decision
        response_time = np.random.normal(45, 15)  # seconds
        
        # Simulate participant decision (for demonstration)
        possible_actions = [
            "open_email",
            "click_link",
            "download_attachment",
            "report_phishing",
            "delete",
            "verify_with_sender",
            "safe_to_open"
        ]
        
        participant_action = random.choice(possible_actions)
        
        # Determine if response was correct
        is_correct = (participant_action == scenario['correct_response'])
        
        result = {
            "scenario_id": scenario['id'],
            "difficulty": scenario['difficulty'],
            "cognitive_trigger": scenario['cognitive_trigger'],
            "participant_action": participant_action,
            "correct_response": scenario['correct_response'],
            "is_correct": is_correct,
            "response_time_seconds": max(5, response_time),  # minimum 5 seconds
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_metrics(self, scenario_results: List[Dict]) -> Dict:
        """Calculate performance metrics from experiment results"""
        total = len(scenario_results)
        correct = sum(1 for r in scenario_results if r['is_correct'])
        
        # Accuracy by difficulty
        accuracy_by_difficulty = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in scenario_results if r['difficulty'] == difficulty]
            if diff_results:
                accuracy_by_difficulty[difficulty] = sum(
                    1 for r in diff_results if r['is_correct']
                ) / len(diff_results)
        
        # Accuracy by cognitive trigger
        accuracy_by_trigger = {}
        for result in scenario_results:
            trigger = result['cognitive_trigger']
            if trigger not in accuracy_by_trigger:
                accuracy_by_trigger[trigger] = []
            accuracy_by_trigger[trigger].append(result['is_correct'])
        
        accuracy_by_trigger = {
            k: sum(v) / len(v) for k, v in accuracy_by_trigger.items()
        }
        
        metrics = {
            "overall_accuracy": correct / total if total > 0 else 0,
            "total_scenarios": total,
            "correct_responses": correct,
            "incorrect_responses": total - correct,
            "average_response_time": np.mean([r['response_time_seconds'] 
                                             for r in scenario_results]),
            "accuracy_by_difficulty": accuracy_by_difficulty,
            "accuracy_by_cognitive_trigger": accuracy_by_trigger
        }
        
        return metrics
    
    def generate_experiment_dataset(self, n_participants: int = 100,
                                   output_path: str = 'data/synthetic/experiment_results.csv'):
        """Generate synthetic experiment dataset"""
        all_results = []
        
        for i in range(n_participants):
            participant_id = f"P{i+1:03d}"
            experiment = self.run_experiment(participant_id)
            
            # Flatten results for CSV
            for scenario_result in experiment['scenarios']:
                row = {
                    'participant_id': participant_id,
                    **scenario_result,
                    **{f"metric_{k}": v for k, v in experiment['performance_metrics'].items() 
                       if not isinstance(v, dict)}
                }
                all_results.append(row)
        
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"Generated experiment data for {n_participants} participants")
        print(f"Saved to {output_path}")
        
        return df


def main():
    """Main function to run experiment simulation"""
    print("Initializing Phishing Scenario Simulator...")
    
    simulator = PhishingScenarioSimulator()
    
    print(f"\nLoaded {len(simulator.scenarios)} phishing scenarios")
    
    # Run sample experiment
    print("\nRunning sample experiment...")
    result = simulator.run_experiment("SAMPLE_001")
    
    print(f"\nSample Experiment Results:")
    print(f"Overall Accuracy: {result['performance_metrics']['overall_accuracy']:.2%}")
    print(f"Average Response Time: {result['performance_metrics']['average_response_time']:.1f} seconds")
    
    # Generate full dataset
    print("\nGenerating full experiment dataset...")
    simulator.generate_experiment_dataset(n_participants=300)
    
    print("\n✓ Experiment simulation complete!")


if __name__ == "__main__":
    main()