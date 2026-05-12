"""
Pytest configuration and shared fixtures for Earth-Agent tests.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
