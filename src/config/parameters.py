"""
Parameters and thresholds
"""

class Parameters:
    """All parameter definitions from the original code"""
    
    @staticmethod
    def get_thresholds():
        return {
            'minimum_confidence': 0.4,
            'good_confidence': 0.65,
            'excellent_confidence': 0.75,
            'minimum_scene_confidence': 0.35,
            'balanced_score_threshold': 0.15,
            'diversity_threshold': 4
        }
    
    @staticmethod
    def get_targets():
        return {
            'max_response_time_ms': 400,
            'target_avg_confidence': 0.75,
            'target_high_confidence_ratio': 0.6,
            'target_category_diversity': 4,
            'max_error_rate': 0.005
        }
    
    @staticmethod
    def get_quality_weights():
        return {
            'base_similarity': 0.45,
            'style_alignment': 0.20,
            'color_harmony': 0.15,
            'material_quality': 0.10,
            'size_appropriateness': 0.05,
            'category_fit': 0.05
        }