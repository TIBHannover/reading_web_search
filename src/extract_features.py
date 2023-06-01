import argparse
import numpy as np
import pandas as pd


class TrackingFeatures:
    def __init__(self, method):
        self.method = method
        self.gaze_data = pd.read_csv(f"data/{method}_regressions.tsv", sep="\t")
        self.features = self.gaze_data[["uid"]].drop_duplicates()
        self.filter_data()

    def run(self):
        self.calculate_features()
        if method == "our":
            self.our_calculate_features()
        self.features = self.features.sort_values(by="uid")

    def save(self):
        self.features = self.features.dropna(subset=["n_RS"])
        print(self.features.describe())
        self.features.to_csv(f"out/{method}_tracking_features.csv", index=False)

    def format_units(self):
        #self.features["sum_fix_dur"] /= (1000*60) # in minutes
        #self.features["mean_dur_per_RS"] /= 1000 # in secs
        #self.features["sum_RF_dur"] /= (1000*60) # in minutes
        self.features["ratio_n_REG_per_sec"] *= 1000 # in secs



    def filter_data(self):
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("google")]
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("youtube")]
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("ecosia")]
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains(".pdf")]
        pass

    def calculate_features(self):
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url"]].drop_duplicates().groupby("uid").count().reset_index().rename(columns={"url": "n_CP_visited"}), on="uid", how="left")

        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "sum_fix_dur"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_fix_dur"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "n_fixs"}), on="uid", how="left")

        # only consider fixations that where classified as reading
        self.gaze_data = self.gaze_data[self.gaze_data["reading_sequence"] > -1]

        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").max().reset_index().rename(columns={"duration": "max_sum_reading_dur_per_content-page"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_sum_reading_dur_per_content-page"}), on="uid", how="left")

        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_dur_per_RS"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence"]].drop_duplicates().groupby(["uid"]).count().reset_index().rename(columns={"reading_sequence": "n_RS"}), on="uid", how="left")

        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "sum_RF_dur"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"duration": "mean_RF_dur_per_CP"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).count().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"duration": "mean_n_RF_per_CP"}), on="uid", how="left")


        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_RF_dur"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "n_RF"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "mean_y"]].groupby("uid").max().reset_index().rename(columns={"mean_y": "max_y_of_RF"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "mean_y"]].groupby("uid").mean().reset_index().rename(columns={"mean_y": "mean_y_of_RF"}), on="uid", how="left")


        self.features["ratio_RF_per_fix"] = self.features["n_RF"] / self.features["n_fixs"]
        self.features["ratio_RF_dur_per_fix_dur"] = self.features["sum_RF_dur"] / self.features["sum_fix_dur"]
        self.features["ratio_n_RF_per_n_RS"] = self.features["n_RF"] / self.features["n_RS"]

        self.features = pd.merge(self.features, self.gaze_data[["uid", "regression"]].groupby("uid").sum().reset_index().rename(columns={"regression": "n_REG"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "reading_sequence"]].drop_duplicates().groupby(["uid", "url"]).count().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"reading_sequence": "mean_n_RS_per_CP"}), on="uid", how="left")
        self.features["ratio_n_REG_per_n_RS"] = self.features["n_REG"] / self.features["n_RS"]
        self.features["ratio_n_REG_per_sec"] =  self.features["n_REG"] / self.features["sum_RF_dur"]
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_RF_in_px"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "sum_len_RS_in_px"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_REG_in_px"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "sum_len_REG_in_px"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "reading_sequence", "distances"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_RS_in_px"}), on="uid", how="left")

        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True].drop_duplicates(subset=["uid", "reading_sequence"], keep="first")[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).min().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_dur_until_first_REG_per_RS"}), on="uid", how="left")


    def our_calculate_features(self):
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "y", "reading_sequence"]].drop_duplicates().groupby(["uid", "url", "y"]).count().reset_index()[["uid", "reading_sequence"]].groupby(["uid"]).mean().reset_index().rename(columns={"reading_sequence": "mean_rows_per_RS"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "eyes_on_word"]].drop_duplicates().groupby("uid").count().reset_index().rename(columns={"eyes_on_word": "session_count_unique_words"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").mean().reset_index().rename(columns={"eyes_on_word": "mean_len_RS_in_idx"}), on="uid", how="left")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").sum().reset_index().rename(columns={"eyes_on_word": "sum_len_RS_in_idx"}), on="uid", how="left")
        self.features["ratio_sum_len_RS_in_idx_per_reading_dur"] = 1000 * self.features["sum_len_RS_in_idx"] / self.features["sum_RF_dur"]

        temp = self.gaze_data[["uid", "eyes_on_word", "regression"]].copy(deep=True)
        temp["jump"] = temp["eyes_on_word"].diff(periods=1).abs()
        self.features = pd.merge(self.features, temp[temp["regression"] == True][["uid", "jump"]].groupby("uid").mean().reset_index().rename(columns={"jump": "mean_len_REG_in_idx"}), on="uid", how="left")
        temp = self.gaze_data[["uid", "url", "reading_sequence", "eyes_on_word"]].copy(deep=True)
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_unique_words_per_url(x)).reset_index().rename(columns={0: "unique_words_read"})
        self.features = pd.merge(self.features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"unique_words_read": "n_unique_words_read"}), on="uid", how="left")
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_words_per_url(x)).reset_index().rename(columns={0: "words_read"})
        self.features = pd.merge(self.features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"words_read": "n_words_read"}), on="uid", how="left")
        self.features["ratio_n_unique_words_per_n_words"] = self.features["n_unique_words_read"] / self.features["n_words_read"]
        self.features["unique_words_per_sec"] = self.features["n_unique_words_read"] / self.features["sum_RF_dur"] * 1000
        self.features["words_per_sec"] = self.features["n_words_read"] / self.features["sum_RF_dur"] * 1000

    def read_unique_words_per_url(self, s):
        p = s.groupby(["uid", "url", "reading_sequence"]).min().reset_index().rename(columns={"eyes_on_word": "min_word_index"})
        p["max_word_index"] = s.groupby(["uid", "url", "reading_sequence"]).max().reset_index()["eyes_on_word"]
        res = []
        for idx, row in p.iterrows():
            a = np.arange(row["min_word_index"], row["max_word_index"]+1, 1, dtype=int)
            res.extend(a)
        return len(list(set(res)))

    def read_words_per_url(self, s):
        p = s.groupby(["uid", "url", "reading_sequence"]).min().reset_index().rename(columns={"eyes_on_word": "min_word_index"})
        p["max_word_index"] = s.groupby(["uid", "url", "reading_sequence"]).max().reset_index()["eyes_on_word"]
        res = []
        for idx, row in p.iterrows():
            a = np.arange(row["min_word_index"], row["max_word_index"]+1, 1, dtype=int)
            res.extend(a)
        return len(res)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="cole2011")
    args = parser.parse_args()
    return args.method

if __name__ == "__main__":
    method = parse_args()
    feature_calculator = TrackingFeatures(method)
    feature_calculator.run()
    feature_calculator.format_units()
    feature_calculator.save()