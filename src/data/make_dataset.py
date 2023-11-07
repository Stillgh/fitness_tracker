import pandas as pd
from glob import glob

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2_MetaWear_2019-01-14T14.27.00"
                              ".784_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
# print(single_file_acc.head())
single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

files = glob("../../data/raw/MetaMotion/*csv")
data_path = "../../data/raw/MetaMotion/"


def read_data(files):
    # --------------------------------------------------------------
    # Read all files
    # --------------------------------------------------------------

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("2").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        elif "Gyroscope" in f:
            df["set"] = acc_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    # --------------------------------------------------------------
    # Working with datetimes
    # --------------------------------------------------------------

    # pd.to_datetime(df["epoch (ms)"], unit="ms")
    # pd.to_datetime(df["time (01:00)"])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# Rename columns
merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

# merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for _, g in merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
data_resampled["set"] = data_resampled["set"].astype("int")
# print(data_resampled.info())



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")

