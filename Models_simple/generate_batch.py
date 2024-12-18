from generate import generate


output_path = "outputs_simple/KV_Merge_25Percent_Mean_OnKey_10layerskip"
base_path = "Models_simple"
class_name = "KV_Merge_25Percent_Mean_OnKey_10layerskip"
dataset_path = "outputs_simple/BaseModel"

generate(output_path, base_path, class_name, dataset_path)