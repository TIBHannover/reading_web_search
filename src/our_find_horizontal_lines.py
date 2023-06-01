import pandas as pd
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time
import multiprocessing
import numpy as np
import os
from glob import glob
from pathlib import Path
from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['config/settings.yaml', 'config/.secrets.yaml'],
)

tqdm.pandas()


# foveal region threshold
# 2° foveal and 7° parafoveal assumed
MIN_FIXATION_TIME_THRESHOLD = settings.eye_tracking_constants.min_fixation_time_threshold
FOVEAL_REGION_RADIUS = settings.eye_tracking_constants.foveal_region_radius
PARAFOVEAL_REGION_RADIUS = settings.eye_tracking_constants.parafoveal_region_radius
ERROR_REGION = settings.our_approach_constants.error_region



class ParallelRunner:
    def __init__(self, uid, gaze_data, words_map):
        self.gaze_data = gaze_data
        self.words_map = words_map
        self.uid = uid

    def run(self):
        if self.uid in self.words_map["uid"].values:
            self.combine_fixations_on_word()
        else:
            print(f"Participant {self.uid} not in words map")

    def point_distance_to_words(self, x, y, df):
        # filtering for less computations
        df = df[~(x < (df["x"]-ERROR_REGION))]
        df = df[~(y < (df["y"]-ERROR_REGION))]
        df = df[~(x > (df["x"]+ERROR_REGION))]
        df = df[~(y > (df["y"]+ERROR_REGION))]

        if df.shape[0] == 0:
            return pd.DataFrame(columns=df.columns)
        df["x_c"] = x
        df.loc[x < df["x"], "x_c"] = df["x"]
        df.loc[x > df["x2"], "x_c"] = df["x2"]

        df["y_c"] = y
        df.loc[y < df["y"], "y_c"] = df["y"]
        df.loc[y > df["y2"], "y_c"] = df["y2"]

        df["distance"] = df.apply(lambda w: euclidean([x, y], [w.x_c, w.y_c]), axis=1)
        min_dists = df[df.distance <= (ERROR_REGION)].sort_values(by="distance")#.iloc[:9]
        return min_dists

    def viterbi(self, words_list):
        try:
            from viterbi_trellis import ViterbiTrellis
            # indeces = [list(zip(list(t.distance.values), list(t.index.values))) for t in words_list]
            indeces = [list(zip([i for i in range(t.shape[0])], list(t.index.values))) for t in words_list]

            # v = ViterbiTrellis(indeces, lambda x: x[0], lambda x, y: 0.5*mean_word_length*(np.abs(1 - y[1] + x[1]))**2) # 0.25*(y[1] - x[1])**2)
            v = ViterbiTrellis(indeces, lambda x: x[0]**2, lambda x, y: (np.abs(1 - y[1] + x[1]))**2)
            best_path = v.viterbi_best_path()
            path = [words_list[i].index.values[idx] for i, idx in enumerate(best_path)]
        except:
            path = [-1 for _ in words_list]
        return path

    def combine_fixations_on_word(self):
        df = self.gaze_data.copy(deep=True)
        new_df = []
        indexe = df.index.values
        if self.uid != 71:
            return False
        else:
            print(71)
        i = 0
        with tqdm(total=df.shape[0]-1) as progress:
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
                near_word = self.point_distance_to_words(cur.mean_x, cur.mean_y, self.words_map[self.words_map["url"] == cur.url])
                nxt_word = self.point_distance_to_words(nxt.mean_x, nxt.mean_y, self.words_map[self.words_map["url"] == nxt.url])

                near_words = [near_word]
                while (self.in_parafoveal_region(cur.mean_x, cur.mean_y, nxt.mean_x, nxt.mean_y) | self.in_foveal_left_region(cur.mean_x, cur.mean_y, nxt.mean_x, nxt.mean_y)) & (len(near_word) > 0) & (len(nxt_word) > 0):
                    temp.append(nxt)
                    near_words.append(nxt_word)
                    temp_df = pd.DataFrame(temp, columns=df.columns)
                    cur = nxt
                    k += 1
                    try:
                        nxt = df.loc[indexe[k]]
                        nxt_word = self.point_distance_to_words(nxt.mean_x, nxt.mean_y, self.words_map[self.words_map["url"] == nxt.url])
                    except IndexError:
                        break

                progress.update(k-i)
                mean_word_length = self.words_map[self.words_map["url"] == cur.url].width.mean()
                vit = self.viterbi(near_words, mean_word_length)
                i = k
                temp_df["eyes_on_word"] = vit
                all_sequences.append(temp_df.copy(deep=True))
                temp = None
        new_df = pd.concat(all_sequences)
        words_map = self.words_map.copy(deep=True).drop(columns=["url", "uid"])
        to_return = pd.merge(new_df, words_map, how="left", left_on=["eyes_on_word"], right_index=True)
        to_return.to_csv(f"data/our_gaze_in_horizontal_line/{self.uid}.tsv", sep="\t", index=False)

    def in_parafoveal_region(self, x0, y0, x1, y1):
        # if point left from reference point, return not in parafoveal region
        if x0 > x1:
            return False

        g_ell_width = PARAFOVEAL_REGION_RADIUS * 2
        g_ell_height = FOVEAL_REGION_RADIUS * 2
        xc = x1 - x0
        yc = y1 - y0
        rad_cc = (xc**2/(g_ell_width/2.)**2) + (yc**2/(g_ell_height/2.)**2)

        return rad_cc <= 1.

    def in_foveal_left_region(self, x0, y0, x1, y1):
        # if point left from reference point, return not in parafoveal region
        if x1 >= x0:
            return False

        g_ell_width = FOVEAL_REGION_RADIUS * 2
        g_ell_height = FOVEAL_REGION_RADIUS * 2
        xc = x1 - x0
        yc = y1 - y0
        rad_cc = (xc**2/(g_ell_width/2.)**2) + (yc**2/(g_ell_height/2.)**2)

        return rad_cc <= 1.


def parallel_proxy(runner):
    runner.run()

if __name__ == "__main__":
    Path("data/our_gaze_in_horizontal_line").mkdir(parents=True, exist_ok=True)
    words_map = pd.read_csv("data/words_map.tsv", sep="\t")
    startTime = time.time()
    result = []
    words_map["x2"] = words_map["x"] + words_map["width"]
    words_map["y2"] = words_map["y"] + words_map["height"]

    parallel_runners = [ParallelRunner(int(os.path.basename(file).replace(".tsv", "")), pd.read_csv(file, sep="\t"), words_map[words_map["uid"] == int(os.path.basename(file).replace(".tsv", ""))].copy(deep=True)) for file in sorted(glob("data/raw_fixations/*.tsv"))]
    cores = multiprocessing.cpu_count() -1
    print(f"num cores: {cores}")
    with multiprocessing.Pool(cores) as pool:
        pool.map(parallel_proxy, parallel_runners)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))