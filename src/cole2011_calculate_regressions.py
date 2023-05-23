import pandas as pd
from scipy.spatial.distance import euclidean
import time
from tqdm import tqdm
tqdm.pandas()

def calculate_regression(s):
    s["regression"] = s["mean_x"].diff().fillna(0) < 0
    return s


def calculate_distances(s):
    distances = [0.0]
    if s.shape[0] > 1:
        indexe = s.index.values
        for i in range(0, len(indexe)-1, 1):
            cur = s.loc[indexe[i]]
            nxt = s.loc[indexe[i+1]]
            distances.append(euclidean([cur.mean_x, cur.mean_y], [nxt.mean_x, nxt.mean_y]))
    s["distances"] = distances
    return s

if __name__ == "__main__":
    in_dir = "data/cole2011_detected_reading_sequences.tsv"
    out_dir = "data/cole2011_regressions.tsv"

    start_time = time.time()
    df = pd.read_csv(in_dir, sep="\t")
    df = df.groupby(["uid", "reading_sequence"], group_keys=False).apply(calculate_distances).reset_index(drop=True)
    df = df.groupby(["uid", "reading_sequence"], group_keys=False).apply(calculate_regression).reset_index(drop=True)
    df.to_csv(out_dir, sep="\t", index=False)
    end_time = time.time()

    print("Runtime: ", end_time - start_time, " seconds")