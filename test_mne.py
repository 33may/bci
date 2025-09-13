#!/usr/bin/env python3
"""Test script for MNE integration with BCI Toolkit."""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bci_toolkit.mne_demo import demo_mne

if __name__ == "__main__":
    print("Testing MNE integration with BCI Toolkit...")
    print(f"Project root: {project_root}")
    
    print("\nStarting BCI Toolkit with MNE integration...")
    print("This will try to load BCI data using mne.datasets.eegbci...")
    demo_mne()