import argparse
import pandas as pd
from pathlib import Path
import os
import soundfile as sf

language_config_name="ny_mw" # ADAPT
datapath=f"{os.getcwd()}/{language_config_name}"

LANG_DIR = Path(datapath)

COLS = [
    "id",
    "file_name",
    "raw_transcription",
    "transcription",
    "tokenized_transcriptions",
    "num_samples",
    "gender",
]

def read_fleurs_tsv(tsv_path):
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            parts = line.split("\t")

            if len(parts) != 7:
                print(f"Skipping malformed line {line_no}: found {len(parts)} fields")
                continue
            rows.append(parts)
    return pd.DataFrame(rows, columns=COLS)

def load_fleurs_split(lang_dir, split_name, tsv_name=None):
    if tsv_name is None:
        tsv_name = split_name

    tsv_path = lang_dir / f"{tsv_name}.tsv"
    audio_dir = lang_dir / split_name

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=COLS,
        encoding="utf-8",
        dtype=str,
        on_bad_lines="skip",
        engine="python",
    )

    # build full audio path
    df["audio"] = df["file_name"].apply(lambda x: str(audio_dir / x))

    # keep only rows whose audio file exists
    df = df[df["audio"].apply(lambda x: Path(x).is_file())].copy()

    # clean / keep useful columns
    df["id"] = df["id"].astype(int)
    df = df[df["gender"].notna()].copy()
    df = df[df["gender"].isin(["MALE", "FEMALE"])].copy()
    gender_map ={
            "MALE": 1,
            "FEMALE": 0,
    }
    df["gender"] = df["gender"].map(gender_map).astype(int)
    df = df.dropna(subset=["gender"]).copy()
    df["gender"] = df["gender"].astype(int)

    df = df[
        [
            "id",
            "audio",
            "transcription",
            "raw_transcription",
            "gender",
        ]
    ]

    # rename for MMS training format
    df = df.rename(columns={"transcription": "sentence"})

    return df

def compute_total_hours(df):
    total_seconds = 0.0

    for audio_path in df["audio"]:
        try:
            with sf.SoundFile(audio_path) as f:
                duration = len(f) / f.samplerate
                total_seconds += duration
        except Exception:
            continue

    total_hours = total_seconds / 3600
    return total_hours

def main():
    parser = argparse.ArgumentParser(description="Preprocess FLEURS dataset")
    
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language directory (e.g., lg_ug)",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scratch/skscla001/experiments/datasets/xtreme_ssa",
        help="Root directory containing language folders",
    )

    args = parser.parse_args()
    
    language_config_name=args.lang

    lang_dir = Path(args.root_dir) / args.lang

    if not lang_dir.exists():
        print(f"ERROR: Language directory does not exist: {lang_dir}")
        sys.exit(1)

    if not lang_dir.is_dir():
        print(f"ERROR: Path exists but is not a directory: {lang_dir}")
        sys.exit(1)

    print("Processing language:", args.lang)
    print(f"Directory: {lang_dir}")

    train_df = load_fleurs_split(lang_dir, "train")
    dev_df   = load_fleurs_split(lang_dir, "dev")
    test_df  = load_fleurs_split(lang_dir, "test")

    train_df.to_csv(f"{lang_dir}/train/train.csv", sep="\t", index=False)
    dev_df.to_csv(f"{lang_dir}/dev/dev.csv", sep="\t", index=False)
    test_df.to_csv(f"{lang_dir}/test/test.csv", sep="\t", index=False)
    test_df.to_csv(f"{lang_dir}/test/combined.csv", sep="\t", index=False)

    # create gender subgroups
    train_male = train_df[train_df["gender"] == 1].copy()
    train_female = train_df[train_df["gender"] == 0].copy()

    dev_male = dev_df[dev_df["gender"] == 1].copy()
    dev_female = dev_df[dev_df["gender"] == 0].copy()

    test_male = test_df[test_df["gender"] == 1].copy()
    test_female = test_df[test_df["gender"] == 0].copy()

    # save gender subgroups
    train_male.to_csv(f"{lang_dir}/train/male.csv",sep="\t", index=False)
    train_female.to_csv(f"{lang_dir}/train/female.csv", sep="\t", index=False)
    
    dev_male.to_csv(f"{lang_dir}/dev/male.csv",sep="\t", index=False)
    dev_female.to_csv(f"{lang_dir}/dev/female.csv", sep="\t", index=False)

    test_male.to_csv(f"{lang_dir}/test/male.csv",sep="\t", index=False)
    test_female.to_csv(f"{lang_dir}/test/female.csv", sep="\t", index=False)
    
    stats_file=f"{lang_dir}/info_stats.txt"
    with open(stats_file, "w") as f:
        f.write(f"\n---- {language_config_name} data stats ----")
        f.write("\ntrain set:" + str(len(train_df)) + "("+str(round(compute_total_hours(train_df),2))+")\n")
        f.write("validation set:" + str(len(dev_df)) + "("+str(round(compute_total_hours(dev_df),2))+")\n")
        f.write("test set:" + str(len(test_df)) + "("+str(round(compute_total_hours(test_df),2))+")\n")
        f.write("\nTrain Subgroup:\n")
        f.write("male set:" + str(len(train_male)) + "("+str(round(compute_total_hours(train_male),2))+")\n")
        f.write("female set:" + str(len(train_female)) + "("+str(round(compute_total_hours(train_female),2))+")\n")
        f.write("\nValidation Subgroup:\n")
        f.write("male set:" + str(len(dev_male)) + "("+str(round(compute_total_hours(dev_male),2))+")\n")
        f.write("female set:" + str(len(dev_female)) + "("+str(round(compute_total_hours(dev_female),2))+")\n")
        f.write("\nTest Subgroup:\n")
        f.write("male set:" + str(len(test_male)) + "("+str(round(compute_total_hours(test_male),2))+")\n")
        f.write("female set:" + str(len(test_female)) + "("+str(round(compute_total_hours(test_female),2))+")\n")

if __name__ == "__main__":
    main()
