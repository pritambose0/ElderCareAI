"""
ElderNest AI - ML Models
=========================

This package contains pre-trained and custom ML models:
- EmotionDetector: DeepFace-based facial emotion detection
- FallDetector: MediaPipe-based fall and posture detection
- ActivityAnalyzer: Activity pattern analysis

These models form the core of the multi-modal risk assessment system.
"""

from app.models.emotion_detector import EmotionDetector, emotion_detector
from app.models.fall_detector import FallDetector, fall_detector
from app.models.activity_analyzer import ActivityAnalyzer, activity_analyzer

__all__ = [
    'EmotionDetector',
    'emotion_detector',
    'FallDetector', 
    'fall_detector',
    'ActivityAnalyzer',
    'activity_analyzer'
]
