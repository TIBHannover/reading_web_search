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

    def save(self):
        print(self.features.describe())
        self.features.to_csv(f"out/{method}_tracking_features.csv", index=False)

    def filter_data(self):
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("google")]
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("youtube")]
        self.gaze_data = self.gaze_data[~self.gaze_data["url"].str.contains("ecosia")]
        pass

    def calculate_features(self):
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "session_sum_fixation_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "session_mean_fixation_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "session_count_fixations"}), on="uid")

        # only consider fixations that where classified as reading
        self.gaze_data = self.gaze_data[self.gaze_data["reading_sequence"] > -1]

        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").max().reset_index().rename(columns={"duration": "content-page_max_sum_reading_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "content-page_mean_sum_reading_duration"}), on="uid")

        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "session_mean_reading_sequence_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence"]].drop_duplicates().groupby(["uid"]).count().reset_index().rename(columns={"reading_sequence": "session_count_reading_sequence"}), on="uid")

        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "session_sum_reading_fixation_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "session_mean_reading_fixation_duration"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "session_count_reading_fixation"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "mean_y"]].groupby("uid").max().reset_index().rename(columns={"mean_y": "session_max_y-position_reading_fixation"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "mean_y"]].groupby("uid").mean().reset_index().rename(columns={"mean_y": "session_mean_y-position_reading_fixation"}), on="uid")


        self.features["ratio_reading_fixations_per_fixation"] = self.features["session_count_reading_fixation"] / self.features["session_count_fixations"]
        self.features["ratio_reading_duration_per_session_duration"] = self.features["session_sum_reading_fixation_duration"] / self.features["session_sum_fixation_duration"]
        self.features["ratio_count_reading_fixations_per_reading_sequence"] = self.features["session_count_reading_fixation"] / self.features["session_count_reading_sequence"]

        self.features = pd.merge(self.features, self.gaze_data[["uid", "regression"]].groupby("uid").sum().reset_index().rename(columns={"regression": "session_count_regressions"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "reading_sequence"]].drop_duplicates().groupby(["uid", "url"]).count().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"reading_sequence": "content-page_mean_reading_sequences"}), on="uid")
        self.features["ratio_count_regressions_per_reading_sequence"] = self.features["session_count_regressions"] / self.features["session_count_reading_sequence"]
        self.features["ratio_reading_time_per_regression"] = self.features["session_sum_reading_fixation_duration"] / self.features["session_count_regressions"]
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "session_mean_len_reading_fixation_in_px"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "session_sum_len_reading_sequences_in_px"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "session_mean_len_regressions_in_px"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "session_sum_len_regressions_in_px"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == False][["uid", "reading_sequence", "distances"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "session_mean_len_reading_sequence_in_px"}), on="uid")

        self.features = pd.merge(self.features, self.gaze_data[self.gaze_data["regression"] == True].drop_duplicates(subset=["uid", "reading_sequence"], keep="first")[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).min().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "session_mean_time_until_first_regression_per_reading_sequence"}), on="uid")

        self.features["ratio_reading_len_in_px_per_reading_time"] = self.features["session_sum_len_reading_sequences_in_px"] / self.features["session_sum_reading_fixation_duration"]


    def our_calculate_features(self):
        self.features = pd.merge(self.features, self.gaze_data[["uid", "url", "y", "reading_sequence"]].drop_duplicates().groupby(["uid", "url", "y"]).count().reset_index()[["uid", "reading_sequence"]].groupby(["uid"]).mean().reset_index().rename(columns={"reading_sequence": "mean_rows_per_reading_sequence"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "eyes_on_word"]].drop_duplicates().groupby("uid").count().reset_index().rename(columns={"eyes_on_word": "session_count_unique_words"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").mean().reset_index().rename(columns={"eyes_on_word": "session_mean_len_reading_sequence_in_indexes"}), on="uid")
        self.features = pd.merge(self.features, self.gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").sum().reset_index().rename(columns={"eyes_on_word": "session_sum_len_reading_sequence_in_indexes"}), on="uid")
        self.features["ratio_reading_len_in_indexes_per_reading_time"] = self.features["session_sum_len_reading_sequences_in_px"] / self.features["session_sum_len_reading_sequence_in_indexes"]

        temp = self.gaze_data[["uid", "eyes_on_word", "regression"]].copy(deep=True)
        temp["jump"] = temp["eyes_on_word"].diff(periods=1).abs()
        self.features = pd.merge(self.features, temp[temp["regression"] == True][["uid", "jump"]].groupby("uid").mean().reset_index().rename(columns={"jump": "session_mean_len_regressions_in_indexes"}), on="uid")
        temp = self.gaze_data[["uid", "url", "reading_sequence", "eyes_on_word"]].copy(deep=True)
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_unique_words_per_url(x)).reset_index().rename(columns={0: "unique_words_read"})
        self.features = pd.merge(self.features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"unique_words_read": "total_unique_words_read"}), on="uid")
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_words_per_url(x)).reset_index().rename(columns={0: "words_read"})
        self.features = pd.merge(self.features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"words_read": "total_words_read"}), on="uid")
        self.features["ratio_unique_words_per_words"] = self.features["total_unique_words_read"] / self.features["total_words_read"]
        self.features["ratio_unique_read_words_per_second"] = self.features["total_unique_words_read"] / self.features["session_sum_reading_fixation_duration"] * 1000

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
    parser.add_argument("--method", default="our")
    args = parser.parse_args()
    return args.method

if __name__ == "__main__":
    method = parse_args()
    feature_calculator = TrackingFeatures(method)
    feature_calculator.run()
    feature_calculator.save()