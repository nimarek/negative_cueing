import pandas as pd
import os

# Define the base directory
base_dir = "/home/exp-psy/Desktop/negative_cueing"

# Define the conditions and phases
conditions = ["positive", "negative", "neutral"]
phases = ["cue", "delay"]

# Function to extract relevant rows and save to file
def extract_and_save(df, condition, phase, output_dir, run):
    # Filter rows based on condition and phase
    filtered = df[df["trial_type"].str.contains(f"{condition}{phase}")]
    
    # Use duration column as modulator for delay period
    if phase == "delay":
        result_df = pd.DataFrame({
            "onset": filtered["onset"],
            "duration": filtered["duration"],
            "modulator": filtered["duration"]  # Use duration as modulator
        })
    else:
        result_df = pd.DataFrame({
            "onset": filtered["onset"],
            "duration": filtered["duration"],
            "response": 1
        })
    
    # Define output file path
    output_file_path = os.path.join(output_dir, f"{condition}_{phase}_run-{run:02d}.txt")
    
    # Save file
    result_df.to_csv(output_file_path, sep=" ", header=False, index=False)

# Function to extract and save baseline events (intertrial/fixation)
def extract_baseline(df, output_dir, run):
    baseline_df = df[df["trial_type"].str.contains("intertrial|fixation")]
    baseline_df = baseline_df.assign(response=1)
    baseline_file_path = os.path.join(output_dir, f"baseline_run-{run:02d}.txt")
    baseline_df[["onset", "duration", "response"]].to_csv(baseline_file_path, sep=" ", header=False, index=False)

# Function to extract and save response events
def extract_responses(df, output_dir, run):
    response_df = df[df["trial_type"].str.contains("response")]
    response_df = response_df.assign(response=1)
    response_file_path = os.path.join(output_dir, f"response_run-{run:02d}.txt")
    response_df[["onset", "duration", "response"]].to_csv(response_file_path, sep=" ", header=False, index=False)

# Loop over subjects and runs
for sub in range(1, 26):
    for run in range(1, 13):
        input_file_path = os.path.join(base_dir, f"sub-{sub:02d}", "func", f"sub-{sub:02d}_task-negativesearcheventrelated_run-{run:02d}_events.tsv")
        output_dir = os.path.join(base_dir, f"sub-{sub:02d}", "func")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_file_path):
            print(f"Input file not found: {input_file_path}")
            continue
        
        df = pd.read_csv(input_file_path, sep="\t")
        
        for condition in conditions:
            for phase in phases:
                extract_and_save(df, condition, phase, output_dir, run)
        
        extract_baseline(df, output_dir, run)
        extract_responses(df, output_dir, run)

print("Files generated successfully!")
