from generate import generate



output_paths = [
    "outputs_simple/Q_Merge_25Percent_Mean"
    "outputs_simple/Q_Merge_25Percent_Mean_2layerskip",
    "outputs_simple/Q_Merge_25Percent_Mean_5layerskip",
    "outputs_simple/Q_Merge_25Percent_Mean_10layerskip",
    "outputs_simple/Q_Merge_25Percent_Mean_15layerskip",
    "outputs_simple/Q_Merge_25Percent_Mean_20layerskip",
    "outputs_simple/Q_Merge_25Percent_Mean_25layerskip",
]

base_path = "Models_simple"

class_names = [
    "Q_Merge_25Percent_Mean",
    "Q_Merge_25Percent_Mean_2layerskip",
    "Q_Merge_25Percent_Mean_5layerskip",
    "Q_Merge_25Percent_Mean_10layerskip",
    "Q_Merge_25Percent_Mean_15layerskip",
    "Q_Merge_25Percent_Mean_20layerskip",
    "Q_Merge_25Percent_Mean_25layerskip",
]

dataset_path = "outputs_simple/BaseModel"



for output_path, class_name in zip(output_paths, class_names):
    generate(output_path, base_path, class_name, dataset_path)