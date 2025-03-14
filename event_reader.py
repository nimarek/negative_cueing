import pandas as pd
import glob
import os
import itertools

root_dir = "/home/data/NegativeCueing_RSA/NegCue_Random"
event_files = sorted(glob.glob(f"{root_dir}/code/events/sub-*/sub-*_task-negativesearch*_run-*_events.tsv"))


for f in event_files:
    filename = os.path.basename(f)
    sub_id = filename.split("_")[0].replace("sub-", "")
    run_id = filename.split("_")[3].replace("run-", "").replace("_events.tsv", "")

    print(f"Processing sub-{sub_id}, run-{run_id}...")
    df = pd.read_csv(f, sep="\t")

    response_mask = df["event_type"].str.contains("response", na=False)
    if response_mask.any():
        response_durations = df.loc[response_mask, "duration"].astype(float)
        demeaned_rt = (response_durations - response_durations.mean()).round(2)
        df.loc[response_mask, "duration"] = demeaned_rt

    onset_fixation = df[df["event_type"].str.startswith("fixation")]["onset"].values
    onset_cue = df[df["event_type"].str.startswith("cue")]["onset"].values
    onset_delay = df[df["event_type"].str.startswith("delay")]["onset"].values
    onset_response = df[response_mask]["onset"].values
    onset_iti = df[df["event_type"].str.startswith("iti")]["onset"].values

    onsets_all = list(filter(None, itertools.chain.from_iterable(itertools.zip_longest(
        onset_fixation, onset_cue, onset_delay, onset_response, onset_iti
    ))))

    dur_fixation = df[df["event_type"].str.startswith("fixation")]["duration"].values
    dur_cue = df[df["event_type"].str.startswith("cue")]["duration"].values
    dur_delay = df[df["event_type"].str.startswith("delay")]["duration"].values
    dur_response = df[response_mask]["duration"].values  # Updated demeaned RT
    dur_iti = df[df["event_type"].str.startswith("iti")]["duration"].values


    duration_all = list(filter(None, itertools.chain.from_iterable(itertools.zip_longest(
        dur_fixation, dur_cue, dur_delay, dur_response, dur_iti
    ))))

    df = df[~df["event_type"].str.contains("search", na=False)]
    df["event_type"] = df["event_type"].str.replace("iti", "intertrial", regex=False)

    trial_type_list = df["trial_type"] + "_" + df["event_type"] + "_" + df["cue_type"]

    merged_df = pd.DataFrame({
        "onset": df["onset"],
        "duration": df["duration"],
        "trial_type": trial_type_list
    })

    output_dir = f"{root_dir}/sub-{sub_id}/func"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/sub-{sub_id}_task-negativesearcheventrelated_run-{run_id}_events.tsv"
    merged_df.to_csv(output_path, index=False, sep="\t")

    print(f"Saved: {output_path}")
