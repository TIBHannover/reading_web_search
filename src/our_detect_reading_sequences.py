import pandas as pd
from tqdm import tqdm
from glob import glob
import numpy as np
from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.yaml', 'config/.secrets.yaml'],
)


MIN_FIXATION_TIME_THRESHOLD = settings.eye_tracking_constants.min_fixation_time_threshold
FOVEAL_REGION_RADIUS = settings.eye_tracking_constants.foveal_region_radius
PARAFOVEAL_REGION_RADIUS = settings.eye_tracking_constants.parafoveal_region_radius
NEXT_WORD_DISTANCE_THRESHOLD = settings.our_approach_constants.next_word_distance_threshold
MIN_WORDS_FOR_READING_SEQUENCE = settings.our_approach_constants.min_words_for_reading_sequence


def in_already_processed_area(df, idx):
    return df.eyes_on_word.min() <= idx <= df.eyes_on_word.max()


def test_already_processed():
    df = pd.DataFrame([3, 5, 7], columns=["eyes_on_word"])
    assert in_already_processed_area(df, 3) == True
    assert in_already_processed_area(df, 5) == True
    assert in_already_processed_area(df, 7) == True
    assert in_already_processed_area(df, 2) == False
    assert in_already_processed_area(df, 8) == False


def word_level(files):
    all_participants = []

    for file in tqdm(files):
        df = pd.read_csv(file, sep="\t")
        df = df[df["duration"] >= MIN_FIXATION_TIME_THRESHOLD].dropna()
        df["reading_sequence"] = -1
        indexe = df.index.values

        i = 0
        with tqdm(total=df.shape[0]-1) as progress:
            sequence_counter = 0
            all_sequences = []
            while i < df.shape[0]:
                current_idx = i
                finished = False
                temp = []
                while not finished:
                    temp.append(df.loc[indexe[current_idx]])
                    if current_idx + 1 >= len(indexe):
                        finished = True
                    elif df.loc[indexe[current_idx + 1], 'eyes_on_word'] - NEXT_WORD_DISTANCE_THRESHOLD <= df.loc[indexe[current_idx], 'eyes_on_word'] <= df.loc[indexe[current_idx + 1], 'eyes_on_word'] or in_already_processed_area(pd.DataFrame(temp, columns=df.columns), df.loc[indexe[current_idx + 1], 'eyes_on_word']):
                        current_idx += 1
                    else:
                        finished = True

                progress.update(current_idx-i)
                i = current_idx + 1

                new_df = pd.DataFrame(temp, columns=df.columns)

                if np.ptp(new_df["eyes_on_word"]) >= MIN_WORDS_FOR_READING_SEQUENCE:
                    new_df["reading_sequence"] = sequence_counter
                    sequence_counter += 1

                all_sequences.append(new_df)

        try:
            all_participants.append(pd.concat(all_sequences))
        except ValueError:
            continue

    all_participants = pd.concat(all_participants)
    all_participants = all_participants.dropna(subset=["reading_sequence"])
    all_participants.to_csv("data/our_detected_reading_sequences.tsv", sep="\t", index=False)


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_already_processed()
    files = glob(f"data/our_gaze_in_horizontal_line/*.tsv")
    word_level(files)
    end_time = time.time()

    print("Runtime: ", end_time - start_time, " seconds")

