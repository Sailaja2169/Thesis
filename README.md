# Phishing Behavioral Cybersecurity Research Project

**The Role of Human Factors in Phishing Attacks: A Behavioral Cybersecurity Study**

## Project Overview

This research project investigates why phishing attacks continue to succeed despite advanced cybersecurity technologies, focusing on human behavioral factors, cognitive biases, and workplace contexts.

**📅 Timeline:** 13-week structured plan (see `plan.jpeg` for visual Gantt chart)  
**🎓 Target:** MSc Thesis Excellence + IEEE Publication  
**📊 Current Progress:** Week 3 of 13 (25% complete)

### Research Questions

1. How do cognitive biases (authority bias, urgency) affect phishing attack susceptibility?
2. To what extent do stress and multitasking increase phishing vulnerability in work contexts?
3. Does previous cybersecurity training reduce working professionals' vulnerability to phishing?
4. How do demographics (age, digital literacy, job role) correlate with phishing vulnerability?

## Key Innovations & Improvements

### 1. **Real-Time Phishing Detection System**
- Browser extension prototype for detecting phishing indicators
- ML-based URL analysis
- User behavior tracking

### 2. **Adaptive Training Platform**
- Personalized training based on individual vulnerability profiles
- Spaced repetition learning algorithms
- Context-aware simulations

### 3. **Comprehensive Analytics Dashboard**
- Real-time vulnerability metrics
- Organizational risk heatmaps
- Longitudinal training effectiveness tracking

### 4. **Advanced ML Models**
- Multi-factor vulnerability prediction
- Cognitive bias pattern recognition
- Contextual risk assessment

## Project Structure

```
.
├── data/                          # Data storage
│   ├── raw/                       # Raw survey/experiment data
│   ├── processed/                 # Cleaned and processed data
│   └── synthetic/                 # Synthetic data for testing
├── src/                           # Source code
│   ├── data_collection/           # Survey and experiment tools
│   ├── analysis/                  # Statistical analysis modules
│   ├── models/                    # ML models
│   ├── visualization/             # Visualization tools
│   ├── simulation/                # Phishing scenario simulations
│   └── utils/                     # Utility functions
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── config/                        # Configuration files
├── results/                       # Analysis results and reports
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the project:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

## Usage

### 1. Data Collection
```bash
python src/data_collection/survey_generator.py
python src/data_collection/experiment_simulator.py
```

### 2. Data Analysis
```bash
# Run statistical analysis
python src/analysis/statistical_analysis.py

# Generate visualizations
python src/visualization/dashboard.py
```

### 3. ML Model Training
```bash
python src/models/train_vulnerability_predictor.py
```

### 4. Interactive Notebooks
```bash
jupyter notebook notebooks/
```

## Key Features

### Data Collection
- Automated survey generation
- Phishing scenario simulations
- Real-time experiment tracking
- Demographic data collection

### Statistical Analysis
- Chi-square tests for categorical associations
- ANOVA for group comparisons
- Regression analysis for predictive modeling
- Correlation analysis

### Machine Learning
- Random Forest vulnerability prediction
- Neural networks for pattern recognition
- Ensemble methods for robust predictions
- Feature importance analysis

### Visualization
- Interactive dashboards
- Risk heatmaps
- Training effectiveness plots
- Demographic correlation charts

## Research Methodology

### Mixed Methods Approach
1. **Quantitative**: Survey data (n=300+ professionals)
2. **Experimental**: Simulated phishing scenarios
3. **Qualitative**: Semi-structured interviews (n=20)

### Data Collection Timeline
- **Phase 1** (Weeks 1-4): Survey deployment and baseline data
- **Phase 2** (Weeks 5-8): Phishing simulations
- **Phase 3** (Weeks 9-12): Training intervention
- **Phase 4** (Weeks 13-16): Follow-up assessment

## Expected Outcomes

1. Identification of high-risk demographic profiles
2. Quantification of cognitive bias impact
3. Training effectiveness metrics
4. Practical organizational recommendations
5. Predictive vulnerability models

## Ethical Considerations

- All research approved by ethics committee
- Informed consent from all participants
- Data anonymization and secure storage
- Debrief sessions post-simulation
- No real credential harvesting

## Contributing

This is a research project. For collaboration inquiries, please contact the project lead.

## License

Academic Use Only - See LICENSE file for details

## Author

**Sailaja Midde**  
MSc in Cybersecurity  
National College of Ireland  

## Acknowledgments

- Research supervisor and ethics committee
- Participating organizations
- Open-source community for tools and libraries

## Citation

If you use this research or methodology, please cite:
```
Midde, S. (2025). The Role of Human Factors in Phishing Attacks: 
A Behavioral Cybersecurity Study. National College of Ireland.
```

## Contact

For questions or collaboration: [Contact via NCI]

---

**Last Updated**: October 2025