#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide a full file path."
    exit 1
fi

# Get the full file path
FILE_PATH="$1"

# Extract the directory, filename, and extension
DIR_PATH=$(dirname "$FILE_PATH")
FILE_NAME=$(basename "$FILE_PATH")
FILE_BASE_NAME="${FILE_NAME%.*}"

# Create the new folder with the same name as the file (without extension)
NEW_FOLDER_PATH="$DIR_PATH/$FILE_BASE_NAME"
mkdir -p "$NEW_FOLDER_PATH/pc"

# Copy the file to the "pc" folder inside the new folder
cp "$FILE_PATH" "$NEW_FOLDER_PATH/pc/"

echo "Folder Structure Created."

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "==================================================="
echo "Generating Dataset..."
conda activate pyoccenv
python "$PARENT_DIR/dataset_generator.py" "$NEW_FOLDER_PATH" "ls3dc" "-t_p" "0" "-op" "99"
echo "Dataset Generated."

echo ""
echo "\n==================================================="
echo "Dividing Dataset..."
conda activate pyoccenv
python "$PARENT_DIR/dataset_divider.py" "$NEW_FOLDER_PATH" "ls3dc" "hpnet" "-crf" "1" "-c" "-a" "-ra" "z" "-vrs" "4" "4" "4" "-trs" "4" "4" "4" "-tnp" "7000" "-vnp" "7000" "-vmnp" "7000" "-tmnp" "7000" -"-input_dataset_folder_name" "dataset" "--output_dataset_folder_name" "dataset_divided_444_333_7k" "-vgs" "3" "3" "3"
echo "Dataset Divided."

# echo "Generating Dataset Validation..."
# python "$PARENT_DIR/dataset_evaluator.py" "$NEW_FOLDER_PATH" "hpnet" "-s" "-p" "--dataset_folder_name" "dataset_divided_222_111_7k" "--data_folder_name" "data" "--result_folder_name" "eval/gt" "--ignore_primitives_orientation" "--unnormalize" "-crf" "1"
# echo "Dataset Validation Generated."

echo ""
echo "==================================================="
echo "HPNet Inference..."
conda activate hpnet
python "$HPNET_DIR/train.py" "--eval" "--data_path=$NEW_FOLDER_PATH/dataset_divided_444_333_7k/hpnet/data" "--vis_dir=$NEW_FOLDER_PATH/dataset_divided_444_333_7k/hpnet/predict" "--log_dir=$NEW_FOLDER_PATH/dataset_divided_444_333_7k/hpnet/predict_log" "--checkpoint_path=$HPNET_DIR/log/train_ls3dc_444_7k/checkpoint.tar" "--val_skip" "1" "--vis"
echo "HPNet Inference Finished."

echo ""
echo "==================================================="
echo "Generating Results Evaluation..."
conda activate pyoccenv
python "$PARENT_DIR/dataset_evaluator.py" "$NEW_FOLDER_PATH" "hpnet" "-s" "-p" "--dataset_folder_name" "dataset_divided_444_333_7k" "--data_folder_name" "predict" "--result_folder_name" "eval/predict" "--ignore_primitives_orientation" "--unnormalize" "-crf" "1"
echo "Dataset Validation Generated."

echo ""
echo "==================================================="
echo "Merging Results..."
conda activate pyoccenv
python "$PARENT_DIR/dataset_merger.py" "$NEW_FOLDER_PATH" "hpnet" "ls3dc" "--input_dataset_folder_name" "dataset_divided_444_333_7k" "--output_dataset_folder_name" "dataset_merged_444_333_7k" "--input_data_folder_name" "predict"
echo "Results Merged."

echo ""
echo "==================================================="
echo "Generating Merged Results Evaluation..."
conda activate pyoccenv
python "$PARENT_DIR/dataset_evaluator.py" "$NEW_FOLDER_PATH" "ls3dc" "-s" "-p" "--dataset_folder_name" "dataset_merged_444_333_7k" "--data_folder_name" "predict" "--result_folder_name" "eval/predict" "--ignore_primitives_orientation" "--unnormalize" "-crf" "1"
echo "Merged Results Evaluation Generated."