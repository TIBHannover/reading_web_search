import pandas as pd
from tqdm import tqdm
from glob import glob
import numpy as np
from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.yaml', 'config/.secrets.yaml'],
)

MIN_FIXATION_COUNT_THRESHOLD = settings.cole2011_constants.min_fixation_count_threshold
MIN_FIXATION_TIME_THRESHOLD = settings.eye_tracking_constants.min_fixation_time_threshold
FOVEAL_REGION_RADIUS = settings.eye_tracking_constants.foveal_region_radius
PARAFOVEAL_REGION_RADIUS = settings.eye_tracking_constants.parafoveal_region_radius


def in_parafoveal_region(x0, y0, x1, y1):
    # if point left from reference point, return not in parafoveal region
    if x0 > x1:
        return False

    g_ell_width = PARAFOVEAL_REGION_RADIUS * 2
    g_ell_height = FOVEAL_REGION_RADIUS * 2
    xc = x1 - x0
    yc = y1 - y0
    rad_cc = (xc**2/(g_ell_width/2.)**2) + (yc**2/(g_ell_height/2.)**2)

    return rad_cc <= 1.


def in_already_processed_area(df, x, y):
    x_min, x_max, y_min, y_max = df.mean_x.min(), df.mean_x.max(), df.mean_y.min(), df.mean_y.max()
    if (y_min <= y <= y_max) & (x_min <= x <= x_max):
        return True
    return False


def test_parafoveal_region():
    x0, y0 = 500, 500
    assert in_parafoveal_region(x0, y0, x0 + 0.1*PARAFOVEAL_REGION_RADIUS, x0+10) == True
    assert in_parafoveal_region(x0, y0, x0 + 0.1*PARAFOVEAL_REGION_RADIUS, x0-10) == True
    assert in_parafoveal_region(x0, y0, x0, y0 + 0.9*FOVEAL_REGION_RADIUS) == True
    assert in_parafoveal_region(x0, y0, x0 + 1.1*PARAFOVEAL_REGION_RADIUS, y0) == False
    assert in_parafoveal_region(x0, y0, x0, 0.9*FOVEAL_REGION_RADIUS) == False
    assert in_parafoveal_region(x0, y0, x0, -0.9*FOVEAL_REGION_RADIUS) == False
    assert in_parafoveal_region(x0, y0, x0-10, y0) == False


def test_already_processed():
    df = pd.DataFrame([[100, 50], [200, 60]], columns=["mean_x", "mean_y"])
    assert in_already_processed_area(df, 150, 55) == True
    assert in_already_processed_area(df, 100, 50) == True
    assert in_already_processed_area(df, 200, 60) == True
    assert in_already_processed_area(df, 90, 55) == False
    assert in_already_processed_area(df, 210, 55) == False
    assert in_already_processed_area(df, 150, 65) == False
    assert in_already_processed_area(df, 150, 45) == False


def cole_2011(files):
    all_participants = []

    for file in tqdm(sorted(files)):
        df = pd.read_csv(file, sep="\t")
        df = df[df["duration"] >= MIN_FIXATION_TIME_THRESHOLD]
        df["reading_sequence"] = -1
        indexe = df.index.values

        i = 0
        with tqdm(total=df.shape[0]-1) as progress:
            sequence_counter = 0
            all_sequences = []
            while i < df.shape[0]:
                cur = df.loc[indexe[i]].copy(deep=True)
                k = i+1
                try:
                    nxt = df.loc[indexe[k]]
                except IndexError:
                    break
                temp = [cur]
                temp_df = pd.DataFrame(temp, columns=df.columns)
                while in_parafoveal_region(cur.mean_x, cur.mean_y, nxt.mean_x, nxt.mean_y) | in_already_processed_area(temp_df, nxt.mean_x, nxt.mean_y):
                    temp.append(nxt)
                    temp_df = pd.DataFrame(temp, columns=df.columns)
                    cur = nxt
                    k += 1
                    try:
                        nxt = df.loc[indexe[k]]
                    except IndexError:
                        break

                progress.update(k-i)
                i = k

                new_df = pd.DataFrame(temp, columns=df.columns)
                if new_df.shape[0] >= MIN_FIXATION_COUNT_THRESHOLD:
                    new_df["reading_sequence"] = sequence_counter
                    sequence_counter += 1
                all_sequences.append(new_df)
                temp = None
        try:
            all_sequences = pd.concat(all_sequences)
            all_participants.append(all_sequences)
        except ValueError:
            continue
    all_participants = pd.concat(all_participants)
    all_participants = all_participants.dropna(subset=["reading_sequence"])
    all_participants.to_csv("data/cole2011_detected_reading_sequences.tsv", sep="\t", index=False)


if __name__ == "__main__":
    test_parafoveal_region()
    test_already_processed()
    files = glob(f"data/raw_fixations/*.tsv")
    cole_2011(files)

