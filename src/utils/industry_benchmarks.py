"""
Industry Phishing Vulnerability Benchmarks
Provides comparison data for risk assessment
"""

class IndustryBenchmarks:
    """
    Industry benchmark data for phishing vulnerability
    Based on research studies and security reports
    """
    
    @staticmethod
    def get_overall_benchmarks():
        """
        Get overall phishing vulnerability benchmarks across industries
        """
        return {
            "average_click_rate": 0.184,  # 18.4% average phishing email click rate
            "average_data_entry_rate": 0.067,  # 6.7% enter credentials
            "average_detection_rate": 0.53,  # 53% of phishing emails are reported
            "vulnerability_score": {
                "low": 0.25,    # 25th percentile
                "median": 0.32, # 50th percentile
                "high": 0.47,   # 75th percentile
            },
            "response_time_seconds": {
                "best": 25,     # Fastest 10%
                "average": 64,  # Mean response time
                "worst": 105,   # Slowest 10%
            }
        }
    
    @staticmethod
    def get_industry_benchmarks():
        """
        Get phishing vulnerability benchmarks by industry
        """
        return {
            "Technology": {
                "click_rate": 0.172,
                "data_entry_rate": 0.048,
                "detection_rate": 0.65,
                "vulnerability_score": 0.29,
            },
            "Financial Services": {
                "click_rate": 0.159,
                "data_entry_rate": 0.051,
                "detection_rate": 0.68,
                "vulnerability_score": 0.27,
            },
            "Healthcare": {
                "click_rate": 0.203,
                "data_entry_rate": 0.072,
                "detection_rate": 0.47,
                "vulnerability_score": 0.35,
            },
            "Education": {
                "click_rate": 0.227,
                "data_entry_rate": 0.081,
                "detection_rate": 0.42,
                "vulnerability_score": 0.39,
            },
            "Manufacturing": {
                "click_rate": 0.198,
                "data_entry_rate": 0.069,
                "detection_rate": 0.51,
                "vulnerability_score": 0.34,
            },
            "Government": {
                "click_rate": 0.184,
                "data_entry_rate": 0.064,
                "detection_rate": 0.55,
                "vulnerability_score": 0.31,
            },
            "Retail": {
                "click_rate": 0.201,
                "data_entry_rate": 0.073,
                "detection_rate": 0.49,
                "vulnerability_score": 0.35,
            },
        }
    
    @staticmethod
    def get_role_benchmarks():
        """
        Get phishing vulnerability benchmarks by job role
        """
        return {
            "IT Professional": {
                "click_rate": 0.143,
                "data_entry_rate": 0.037,
                "detection_rate": 0.73,
                "vulnerability_score": 0.24,
            },
            "Executive": {
                "click_rate": 0.187,
                "data_entry_rate": 0.069,
                "detection_rate": 0.58,
                "vulnerability_score": 0.31,
            },
            "Manager": {
                "click_rate": 0.172,
                "data_entry_rate": 0.059,
                "detection_rate": 0.61,
                "vulnerability_score": 0.29,
            },
            "Administrative": {
                "click_rate": 0.209,
                "data_entry_rate": 0.078,
                "detection_rate": 0.46,
                "vulnerability_score": 0.36,
            },
            "Sales": {
                "click_rate": 0.219,
                "data_entry_rate": 0.082,
                "detection_rate": 0.44,
                "vulnerability_score": 0.38,
            },
            "Finance": {
                "click_rate": 0.163,
                "data_entry_rate": 0.054,
                "detection_rate": 0.65,
                "vulnerability_score": 0.28,
            },
            "HR": {
                "click_rate": 0.195,
                "data_entry_rate": 0.071,
                "detection_rate": 0.52,
                "vulnerability_score": 0.33,
            },
            "Other": {
                "click_rate": 0.201,
                "data_entry_rate": 0.073,
                "detection_rate": 0.49,
                "vulnerability_score": 0.35,
            },
        }
    
    @staticmethod
    def get_age_benchmarks():
        """
        Get phishing vulnerability benchmarks by age group
        """
        return {
            "18-25": {
                "click_rate": 0.214,
                "data_entry_rate": 0.079,
                "detection_rate": 0.45,
                "vulnerability_score": 0.37,
            },
            "26-35": {
                "click_rate": 0.181,
                "data_entry_rate": 0.063,
                "detection_rate": 0.57,
                "vulnerability_score": 0.31,
            },
            "36-45": {
                "click_rate": 0.176,
                "data_entry_rate": 0.061,
                "detection_rate": 0.59,
                "vulnerability_score": 0.30,
            },
            "46-55": {
                "click_rate": 0.183,
                "data_entry_rate": 0.065,
                "detection_rate": 0.56,
                "vulnerability_score": 0.31,
            },
            "56+": {
                "click_rate": 0.219,
                "data_entry_rate": 0.083,
                "detection_rate": 0.43,
                "vulnerability_score": 0.38,
            },
        }
    
    @staticmethod
    def get_training_benchmarks():
        """
        Get phishing vulnerability benchmarks by training recency
        """
        return {
            "No Training": {
                "click_rate": 0.276,
                "data_entry_rate": 0.112,
                "detection_rate": 0.31,
                "vulnerability_score": 0.47,
            },
            "Within 3 months": {
                "click_rate": 0.143,
                "data_entry_rate": 0.037,
                "detection_rate": 0.74,
                "vulnerability_score": 0.24,
            },
            "3-6 months": {
                "click_rate": 0.168,
                "data_entry_rate": 0.054,
                "detection_rate": 0.63,
                "vulnerability_score": 0.29,
            },
            "6-12 months": {
                "click_rate": 0.197,
                "data_entry_rate": 0.071,
                "detection_rate": 0.52,
                "vulnerability_score": 0.34,
            },
            "Over 1 year": {
                "click_rate": 0.237,
                "data_entry_rate": 0.094,
                "detection_rate": 0.37,
                "vulnerability_score": 0.42,
            },
        }
    
    @staticmethod
    def get_workload_benchmarks():
        """
        Get phishing vulnerability benchmarks by workload/stress
        """
        return {
            "Low Stress": {
                "click_rate": 0.153,
                "data_entry_rate": 0.043,
                "detection_rate": 0.69,
                "vulnerability_score": 0.26,
            },
            "Medium Stress": {
                "click_rate": 0.184,
                "data_entry_rate": 0.067,
                "detection_rate": 0.53,
                "vulnerability_score": 0.32,
            },
            "High Stress": {
                "click_rate": 0.231,
                "data_entry_rate": 0.089,
                "detection_rate": 0.41,
                "vulnerability_score": 0.40,
            },
        }
    
    @staticmethod 
    def get_benchmark_by_profile(profile):
        """
        Get closest benchmark for a user profile
        
        Args:
            profile: Dictionary with user attributes
            
        Returns:
            Benchmark data for comparison
        """
        benchmarks = {}
        
        # Get industry benchmark
        if "industry" in profile:
            industry_data = IndustryBenchmarks.get_industry_benchmarks()
            if profile["industry"] in industry_data:
                benchmarks["industry"] = industry_data[profile["industry"]]
            else:
                # Default to overall average
                overall = IndustryBenchmarks.get_overall_benchmarks()
                benchmarks["industry"] = {
                    "click_rate": overall["average_click_rate"],
                    "data_entry_rate": overall["average_data_entry_rate"],
                    "detection_rate": overall["average_detection_rate"],
                    "vulnerability_score": overall["vulnerability_score"]["median"],
                }
        
        # Get role benchmark
        if "job_role" in profile:
            role_data = IndustryBenchmarks.get_role_benchmarks()
            if profile["job_role"] in role_data:
                benchmarks["role"] = role_data[profile["job_role"]]
            else:
                benchmarks["role"] = role_data["Other"]
        
        # Get age benchmark
        if "age" in profile:
            age = profile["age"]
            age_data = IndustryBenchmarks.get_age_benchmarks()
            
            if age <= 25:
                benchmarks["age"] = age_data["18-25"]
            elif age <= 35:
                benchmarks["age"] = age_data["26-35"]
            elif age <= 45:
                benchmarks["age"] = age_data["36-45"]
            elif age <= 55:
                benchmarks["age"] = age_data["46-55"]
            else:
                benchmarks["age"] = age_data["56+"]
        
        # Get training benchmark
        if "has_training" in profile:
            training_data = IndustryBenchmarks.get_training_benchmarks()
            
            if not profile["has_training"]:
                benchmarks["training"] = training_data["No Training"]
            elif "training_recency" in profile:
                recency = profile["training_recency"]
                
                if recency == "This week" or recency == "This month":
                    benchmarks["training"] = training_data["Within 3 months"]
                elif recency == "3-6 months ago":
                    benchmarks["training"] = training_data["3-6 months"]
                elif recency == "6-12 months ago":
                    benchmarks["training"] = training_data["6-12 months"]
                else:
                    benchmarks["training"] = training_data["Over 1 year"]
        
        # Get stress/workload benchmark
        if "stress_level" in profile or "workload" in profile:
            stress = profile.get("stress_level", 3)
            workload = profile.get("workload", 3)
            
            # Average stress and workload
            avg_stress = (stress + workload) / 2
            
            stress_data = IndustryBenchmarks.get_workload_benchmarks()
            
            if avg_stress < 2.5:
                benchmarks["stress"] = stress_data["Low Stress"]
            elif avg_stress < 4:
                benchmarks["stress"] = stress_data["Medium Stress"]
            else:
                benchmarks["stress"] = stress_data["High Stress"]
        
        # Overall comparison
        overall = IndustryBenchmarks.get_overall_benchmarks()
        benchmarks["overall"] = {
            "click_rate": overall["average_click_rate"],
            "data_entry_rate": overall["average_data_entry_rate"],
            "detection_rate": overall["average_detection_rate"],
            "vulnerability_score": overall["vulnerability_score"]["median"],
        }
        
        return benchmarks