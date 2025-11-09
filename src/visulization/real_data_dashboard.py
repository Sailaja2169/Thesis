"""
Interactive Dashboard for Real Phishing Research Data
Visualizes actual survey responses from participants
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


class RealDataDashboard:
    """
    Dashboard for visualizing real survey data
    """
    
    def __init__(self):
        st.set_page_config(
            page_title="Phishing Study - Real Data",
            page_icon="🔒",
            layout="wide"
        )
        
        self.data = None
        self.summary = None
    
    def load_data(self):
        """Load processed real data"""
        try:
            self.data = pd.read_csv('data/processed/real_survey_data_cleaned.csv')
            with open('data/processed/real_data_summary_report.json', 'r') as f:
                self.summary = json.load(f)
            return True
        except FileNotFoundError:
            st.error("⚠️ Processed data not found. Please run real_data_processor.py first.")
            return False
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🔒 Phishing Behavioral Study - Real Data Analysis")
        st.markdown("**Sailaja Midde** | MSc Cybersecurity | National College of Ireland")
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Participants", len(self.data))
        col2.metric("Avg Phishing Confidence", f"{self.data['phishing_confidence'].mean():.2f}/5")
        col3.metric("With Training", f"{self.data['prior_training'].sum()}")
        col4.metric("Avg Vulnerability Score", f"{self.data['vulnerability_score'].mean():.2f}/5")
    
    def render_research_questions(self):
        """Display research questions being addressed"""
        st.sidebar.markdown("## 🎯 Research Questions")
        st.sidebar.markdown("""
        **RQ1:** How do authority/urgency cues affect phishing susceptibility?
        
        **RQ2:** Does stress/multitasking increase vulnerability?
        
        **RQ3:** Does prior training reduce vulnerability?
        
        **RQ4:** Do demographics correlate with phishing risk?
        """)
    
    def render_demographics(self):
        """Visualize demographics"""
        st.subheader("👥 Participant Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_counts = self.data['age_bracket'].value_counts()
            fig = px.pie(
                values=age_counts.values,
                names=age_counts.index,
                title="Age Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training status
            training_counts = self.data['prior_training'].map({0: 'No Training', 1: 'Has Training'}).value_counts()
            fig = px.pie(
                values=training_counts.values,
                names=training_counts.index,
                title="Cybersecurity Training Status",
                color_discrete_map={'Has Training': '#2ecc71', 'No Training': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Job roles
        st.markdown("### 💼 Job Role Distribution")
        job_counts = self.data['job_role'].value_counts().head(10)
        fig = px.bar(
            x=job_counts.values,
            y=job_counts.index,
            orientation='h',
            title="Top Job Roles in Study",
            labels={'x': 'Count', 'y': 'Job Role'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_training_analysis(self):
        """RQ3: Analyze training effectiveness"""
        st.subheader("📚 RQ3: Training Effectiveness Analysis")
        
        # Compare key metrics by training status
        trained = self.data[self.data['prior_training'] == 1]
        untrained = self.data[self.data['prior_training'] == 0]
        
        metrics = {
            'Phishing Confidence': 'phishing_confidence',
            'Digital Literacy': 'digital_literacy',
            'Vulnerability Score': 'vulnerability_score'
        }
        
        comparison_data = []
        for metric_name, metric_col in metrics.items():
            comparison_data.append({
                'Metric': metric_name,
                'With Training': trained[metric_col].mean(),
                'No Training': untrained[metric_col].mean()
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='With Training',
            x=df_comparison['Metric'],
            y=df_comparison['With Training'],
            marker_color='#2ecc71'
        ))
        fig.add_trace(go.Bar(
            name='No Training',
            x=df_comparison['Metric'],
            y=df_comparison['No Training'],
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Key Metrics Comparison: Training vs No Training",
            barmode='group',
            yaxis_title="Score (1-5)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance
        from scipy.stats import ttest_ind
        
        st.markdown("#### Statistical Analysis")
        
        for metric_name, metric_col in metrics.items():
            t_stat, p_value = ttest_ind(
                trained[metric_col].dropna(),
                untrained[metric_col].dropna()
            )
            
            if p_value < 0.05:
                sig_text = f"✓ **Significant difference** (p = {p_value:.4f})"
                color = "green"
            else:
                sig_text = f"✗ No significant difference (p = {p_value:.4f})"
                color = "orange"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.markdown(f"**{metric_name}**")
            col2.metric("t-statistic", f"{t_stat:.2f}")
            col3.markdown(f":{color}[{sig_text}]")
    
    def render_stress_multitasking_analysis(self):
        """RQ2: Analyze stress and multitasking impact"""
        st.subheader("😰 RQ2: Stress & Multitasking Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress vs Vulnerability
            fig = px.scatter(
                self.data,
                x='stress_level',
                y='vulnerability_score',
                size='phishing_confidence',
                color='prior_training',
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                title="Stress Level vs Vulnerability Score",
                labels={
                    'stress_level': 'Stress Level (1-5)',
                    'vulnerability_score': 'Vulnerability Score',
                    'prior_training': 'Training Status'
                },
                hover_data=['job_role', 'digital_literacy']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Multitasking frequency distribution
            multitask_order = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']
            multitask_data = self.data.groupby('multitasking')['vulnerability_score'].mean().reindex(multitask_order)
            
            fig = px.bar(
                x=multitask_data.index,
                y=multitask_data.values,
                title="Vulnerability by Multitasking Frequency",
                labels={'x': 'Multitasking Frequency', 'y': 'Avg Vulnerability Score'},
                color=multitask_data.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("#### Correlation Analysis")
        
        from scipy.stats import pearsonr
        
        correlations = {
            'Stress → Vulnerability': ('stress_level', 'vulnerability_score'),
            'Multitasking → Vulnerability': ('multitasking_score', 'vulnerability_score'),
            'Stress → Phishing Confidence': ('stress_level', 'phishing_confidence')
        }
        
        for label, (var1, var2) in correlations.items():
            r, p = pearsonr(self.data[var1].dropna(), self.data[var2].dropna())
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.markdown(f"**{label}**")
            col2.metric("Correlation (r)", f"{r:.3f}")
            
            if abs(r) > 0.3:
                strength = "Strong" if abs(r) > 0.5 else "Moderate"
                direction = "positive" if r > 0 else "negative"
                col3.markdown(f":{('green' if direction=='negative' else 'red')}[{strength} {direction}]")
            else:
                col3.markdown(":gray[Weak correlation]")
    
    def render_authority_bias_analysis(self):
        """RQ1: Analyze authority bias"""
        st.subheader("👔 RQ1: Authority Bias Analysis")
        
        # Authority bias distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                self.data,
                x='authority_bias',
                nbins=5,
                title="Authority Bias Score Distribution",
                labels={'authority_bias': 'Authority Bias (1-5)', 'count': 'Number of Participants'},
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Authority bias vs vulnerability
            fig = px.scatter(
                self.data,
                x='authority_bias',
                y='vulnerability_score',
                size='stress_level',
                color='phishing_confidence',
                title="Authority Bias vs Vulnerability",
                labels={
                    'authority_bias': 'Authority Bias Score',
                    'vulnerability_score': 'Vulnerability Score',
                    'stress_level': 'Stress Level',
                    'phishing_confidence': 'Phishing Confidence'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insight
        avg_authority = self.data['authority_bias'].mean()
        if avg_authority > 3.5:
            st.warning(f"⚠️ **High Authority Bias Detected**: Average score of {avg_authority:.2f}/5 suggests participants may be vulnerable to authority-based phishing attacks.")
        else:
            st.info(f"ℹ️ **Moderate Authority Bias**: Average score of {avg_authority:.2f}/5")
    
    def render_risk_assessment(self):
        """Display risk levels"""
        st.subheader("⚠️ Individual Risk Assessment")
        
        # Risk level distribution
        risk_counts = self.data['risk_level'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Detailed participant table
            st.markdown("#### Participant Risk Profiles")
            display_data = self.data[[
                'participant_id', 'job_role', 'phishing_confidence',
                'digital_literacy', 'vulnerability_score', 'risk_level'
            ]].sort_values('vulnerability_score', ascending=False)
            
            st.dataframe(
                display_data.style.background_gradient(subset=['vulnerability_score'], cmap='Reds'),
                use_container_width=True
            )
    
    def render_recommendations(self):
        """Generate recommendations based on data"""
        st.subheader("💡 Key Findings & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Key Findings")
            
            # Finding 1: Training impact
            trained_confidence = self.data[self.data['prior_training']==1]['phishing_confidence'].mean()
            untrained_confidence = self.data[self.data['prior_training']==0]['phishing_confidence'].mean()
            
            st.markdown(f"""
            **1. Training Impact**
            - Participants with training: {trained_confidence:.2f}/5 confidence
            - Participants without training: {untrained_confidence:.2f}/5 confidence
            - Difference: {(trained_confidence - untrained_confidence):.2f} points
            
            **2. Authority Bias**
            - Average authority bias: {self.data['authority_bias'].mean():.2f}/5
            - {(self.data['authority_bias'] >= 4).sum()} participants show high susceptibility
            
            **3. Stress & Multitasking**
            - Average stress level: {self.data['stress_level'].mean():.2f}/5
            - Most common multitasking: {self.data['multitasking'].mode()[0]}
            
            **4. Overall Vulnerability**
            - Medium risk: {(self.data['risk_level']=='Medium').sum()} participants
            - High risk: {(self.data['risk_level']=='High').sum()} participants
            """)
        
        with col2:
            st.markdown("### 📋 Recommendations")
            st.markdown("""
            **Immediate Actions:**
            1. **Targeted Training**: Focus on participants without prior training
            2. **Authority Awareness**: Emphasize verification of authority-based requests
            3. **Stress Management**: Provide training during low-stress periods
            4. **Multitasking Education**: Encourage focused email review
            
            **Long-term Strategies:**
            5. **Regular Reinforcement**: Schedule quarterly refresher training
            6. **Simulated Phishing**: Implement ongoing testing campaigns
            7. **Just-in-Time Feedback**: Deploy real-time email warnings
            8. **Role-Based Training**: Customize content for job roles
            
            **Research Extensions:**
            9. **Longitudinal Follow-up**: Track training decay over time
            10. **Expanded Sample**: Recruit more participants for statistical power
            """)
    
    def run(self):
        """Main dashboard execution"""
        if not self.load_data():
            st.stop()
        
        # Sidebar navigation
        st.sidebar.title("📊 Navigation")
        self.render_research_questions()
        
        page = st.sidebar.radio(
            "Select View",
            [
                "📊 Overview",
                "👥 Demographics",
                "📚 RQ3: Training Analysis",
                "😰 RQ2: Stress & Multitasking",
                "👔 RQ1: Authority Bias",
                "⚠️ Risk Assessment",
                "💡 Recommendations"
            ]
        )
        
        # Render header on all pages
        self.render_header()
        
        # Render selected page
        if page == "📊 Overview":
            self.render_demographics()
            self.render_risk_assessment()
        elif page == "👥 Demographics":
            self.render_demographics()
        elif page == "📚 RQ3: Training Analysis":
            self.render_training_analysis()
        elif page == "😰 RQ2: Stress & Multitasking":
            self.render_stress_multitasking_analysis()
        elif page == "👔 RQ1: Authority Bias":
            self.render_authority_bias_analysis()
        elif page == "⚠️ Risk Assessment":
            self.render_risk_assessment()
        elif page == "💡 Recommendations":
            self.render_recommendations()


def main():
    """Launch dashboard"""
    dashboard = RealDataDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()