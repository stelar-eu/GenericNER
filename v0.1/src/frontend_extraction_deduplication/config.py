import os

# config.py

# Define paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_CSV_PATH = os.path.join(BASE_DIR, "annotations_df_expanded.csv")
CANDIDATE_PAIRS_CSV_PATH = os.path.join(BASE_DIR, "candidate_pairs_no_dups_augmented.csv")