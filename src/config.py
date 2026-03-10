import os

# Base project directory (parent of src/)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# Output paths
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Matches metadata file
MATCHES_FILE = os.path.join(PROJECT_DIR, "data", "106.json")

# Feature-building parameters
LOOKBACK = 5       # minutes of rolling window
MAX_MINUTE = 85    # last minute to build windows for
MIN_MINUTE = 5     # first minute (lookback starts at 0)
MATCH_END = 90     # match duration for simulation

# Bootstrap parameters
N_BOOT = 1000
BOOT_SEED = 42
