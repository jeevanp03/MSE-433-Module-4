"""
Shared visualization utilities and constants.
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress specific expected warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Physician color palette
PHYS_COLORS = {
    "Dr. A": "#2196F3",
    "Dr. B": "#FF5722",
    "Dr. C": "#4CAF50",
}

# Class colors for outlier visualizations
CLASS_COLORS = {0: "steelblue", 1: "orange", 2: "red"}

# Default DPI for saved figures
SAVE_DPI = 150
