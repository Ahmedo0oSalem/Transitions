import sys
sys.path.insert(0, '.')
from Transitions.analytics.pitch_control import compute_pitch_control_for_match
from Transitions.io.paths import PROCESSED_DIR

if __name__ == "__main__":
    # Change match_id to a processed match you have
    match_id = '3812'
    print(f"Computing pitch control for match {match_id}...")
    df = compute_pitch_control_for_match(match_id, processed_dir=str(PROCESSED_DIR))
    print(df.head(10))
