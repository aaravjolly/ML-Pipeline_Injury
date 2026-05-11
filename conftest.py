"""Pytest configuration: ensure project root is on the import path."""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Silence noisy sklearn deprecation warnings that aren't from our code.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
