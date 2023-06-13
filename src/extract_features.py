import argparse
import numpy as np
import pandas as pd


class TrackingFeatures:
    def __init__(self, method):
        self.method = method
        self.gaze_data = pd.read_csv(f"data/{method}_regressions.tsv", sep="\t")

    def run(self, data):
        features = self.calculate_features(data)
        if method == "our":
            features = self.our_calculate_features(features, data)
        features = features.sort_values(by="uid")
        return features

    def save(self, data, name):
        data = data.dropna(subset=["n_RS"])
        data.to_csv(f"out/{self.method}_{name}.csv", index=False)

    def format_units(self, data):
        #self.features["sum_fix_dur"] /= (1000*60) # in minutes
        #self.features["mean_dur_per_RS"] /= 1000 # in secs
        #self.features["sum_RF_dur"] /= (1000*60) # in minutes
        data["ratio_n_REG_per_sec"] *= 1000 # in secs
        return data

    def filter_serps(self):
        data = self.gaze_data[self.gaze_data["url"].str.contains("google")].copy(deep=True)
        data = data[data["url"].str.contains("search")]
        data = data[~data["url"].str.contains("tbm=vid")]
        data = data[~data["url"].str.contains("tbm=isch")]
        return data

    def filter_content_pages(self):
        data = self.gaze_data[~self.gaze_data["url"].str.contains("google")].copy(deep=True)
        data = data[~data["url"].str.contains("youtube")]
        data = data[~data["url"].str.contains("ecosia")]
        data = data[~data["url"].str.contains(".pdf")]
        return data

    def calculate_features(self, gaze_data):
        features = self.gaze_data[["uid"]].drop_duplicates()
        features = pd.merge(features, gaze_data[["uid", "url"]].drop_duplicates().groupby("uid").count().reset_index().rename(columns={"url": "n_CP_visited"}), on="uid", how="left")

        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "sum_fix_dur"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_fix_dur"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "n_fixs"}), on="uid", how="left")

        # only consider fixations that where classified as reading
        gaze_data = gaze_data[gaze_data["reading_sequence"] > -1]

        features = pd.merge(features, gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").max().reset_index().rename(columns={"duration": "max_sum_reading_dur_per_content-page"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_sum_reading_dur_per_content-page"}), on="uid", how="left")

        features = pd.merge(features, gaze_data[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_dur_per_RS"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "reading_sequence"]].drop_duplicates().groupby(["uid"]).count().reset_index().rename(columns={"reading_sequence": "n_RS"}), on="uid", how="left")

        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").sum().reset_index().rename(columns={"duration": "sum_RF_dur"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).sum().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"duration": "mean_RF_dur_per_CP"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "url", "duration"]].groupby(["uid", "url"]).count().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"duration": "mean_n_RF_per_CP"}), on="uid", how="left")


        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_RF_dur"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "duration"]].groupby("uid").count().reset_index().rename(columns={"duration": "n_RF"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "mean_y"]].groupby("uid").max().reset_index().rename(columns={"mean_y": "max_y_of_RF"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "mean_y"]].groupby("uid").mean().reset_index().rename(columns={"mean_y": "mean_y_of_RF"}), on="uid", how="left")


        features["ratio_RF_per_fix"] = features["n_RF"] / features["n_fixs"]
        features["ratio_RF_dur_per_fix_dur"] = features["sum_RF_dur"] / features["sum_fix_dur"]
        features["ratio_n_RF_per_n_RS"] = features["n_RF"] / features["n_RS"]

        features = pd.merge(features, gaze_data[["uid", "regression"]].groupby("uid").sum().reset_index().rename(columns={"regression": "n_REG"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "url", "reading_sequence"]].drop_duplicates().groupby(["uid", "url"]).count().reset_index().groupby(["uid"]).mean(numeric_only=True).reset_index().rename(columns={"reading_sequence": "mean_n_RS_per_CP"}), on="uid", how="left")
        features["ratio_n_REG_per_n_RS"] = features["n_REG"] / features["n_RS"]
        features["ratio_n_REG_per_sec"] =  features["n_REG"] / features["sum_RF_dur"]
        features = pd.merge(features, gaze_data[gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_RF_in_px"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[gaze_data["regression"] == False][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "sum_len_RS_in_px"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_REG_in_px"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[gaze_data["regression"] == True][["uid", "distances"]].groupby("uid").sum().reset_index().rename(columns={"distances": "sum_len_REG_in_px"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[gaze_data["regression"] == False][["uid", "reading_sequence", "distances"]].groupby(["uid", "reading_sequence"]).sum().reset_index()[["uid", "distances"]].groupby("uid").mean().reset_index().rename(columns={"distances": "mean_len_RS_in_px"}), on="uid", how="left")

        features = pd.merge(features, gaze_data[gaze_data["regression"] == True].drop_duplicates(subset=["uid", "reading_sequence"], keep="first")[["uid", "reading_sequence", "duration"]].groupby(["uid", "reading_sequence"]).min().reset_index()[["uid", "duration"]].groupby("uid").mean().reset_index().rename(columns={"duration": "mean_dur_until_first_REG_per_RS"}), on="uid", how="left")

        return features


    def our_calculate_features(self, features, gaze_data):
        # only consider fixations that where classified as reading
        gaze_data = gaze_data[gaze_data["reading_sequence"] > -1]

        features = pd.merge(features, gaze_data[["uid", "url", "y", "reading_sequence"]].drop_duplicates().groupby(["uid", "url", "y"]).count().reset_index()[["uid", "reading_sequence"]].groupby(["uid"]).mean().reset_index().rename(columns={"reading_sequence": "mean_rows_per_RS"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "eyes_on_word"]].drop_duplicates().groupby("uid").count().reset_index().rename(columns={"eyes_on_word": "session_count_unique_words"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").mean().reset_index().rename(columns={"eyes_on_word": "mean_len_RS_in_idx"}), on="uid", how="left")
        features = pd.merge(features, gaze_data[["uid", "reading_sequence", "eyes_on_word"]].groupby(["uid", "reading_sequence"]).agg(np.ptp).reset_index()[["uid", "eyes_on_word"]].groupby("uid").sum().reset_index().rename(columns={"eyes_on_word": "sum_len_RS_in_idx"}), on="uid", how="left")
        features["ratio_sum_len_RS_in_idx_per_reading_dur"] = 1000 * features["sum_len_RS_in_idx"] / features["sum_RF_dur"]

        temp = gaze_data[["uid", "eyes_on_word", "regression"]].copy(deep=True)
        temp["jump"] = temp["eyes_on_word"].diff(periods=1).abs()
        features = pd.merge(features, temp[temp["regression"] == True][["uid", "jump"]].groupby("uid").mean().reset_index().rename(columns={"jump": "mean_len_REG_in_idx"}), on="uid", how="left")
        temp = gaze_data[["uid", "url", "reading_sequence", "eyes_on_word"]].copy(deep=True)
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_unique_words_per_url(x)).reset_index().rename(columns={0: "unique_words_read"})
        features = pd.merge(features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"unique_words_read": "n_unique_words_read"}), on="uid", how="left")
        t = temp[["uid", "url", "reading_sequence", "eyes_on_word"]].groupby(["uid", "url"]).apply(lambda x: self.read_words_per_url(x)).reset_index().rename(columns={0: "words_read"})
        features = pd.merge(features, t.groupby(["uid"]).sum(numeric_only=True).reset_index().rename(columns={"words_read": "n_words_read"}), on="uid", how="left")
        features["ratio_n_unique_words_per_n_words"] = features["n_unique_words_read"] / features["n_words_read"]
        features["unique_words_per_sec"] = features["n_unique_words_read"] / features["sum_RF_dur"] * 1000
        features["words_per_sec"] = features["n_words_read"] / features["sum_RF_dur"] * 1000

        return features

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
    whole_data = feature_calculator.gaze_data
    overall_features = feature_calculator.run(whole_data)
    serp_data = feature_calculator.filter_serps()
    serp_features = feature_calculator.run(serp_data)
    content_page_data = feature_calculator.filter_content_pages()
    content_page_features = feature_calculator.run(content_page_data)

    overall_features = feature_calculator.format_units(overall_features)
    serp_features = feature_calculator.format_units(serp_features)
    content_page_features = feature_calculator.format_units(content_page_features)

    feature_calculator.save(overall_features, "overall_features")
    feature_calculator.save(serp_features, "serp_features")
    feature_calculator.save(content_page_features, "content_page_features")