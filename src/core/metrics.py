from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    avg_confidence: float
    high_confidence_ratio: float
    category_diversity: int
    processing_time_ms: float
    quality_score: float
    timestamp: str