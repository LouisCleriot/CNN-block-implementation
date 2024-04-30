#!/bin/bash
#how to use, run the following command :
#chmod +x split_dataset.sh
#./split_dataset.sh /path/to/your/dataset 80

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 path_to_dataset train_split_percentage"
    exit 1
fi

DATASET_PATH="$1"
TRAIN_SPLIT="$2"

# Check if the training split percentage is a valid number
if ! [[ "$TRAIN_SPLIT" =~ ^[0-9]+$ ]] || [ "$TRAIN_SPLIT" -lt 0 ] || [ "$TRAIN_SPLIT" -gt 100 ]; then
    echo "The training split percentage must be a number between 0 and 100."
    exit 1
fi

# Create the Train and Test directories
mkdir -p "${DATASET_PATH}/Train"
mkdir -p "${DATASET_PATH}/Test"

# Loop through each directory (class) in the dataset
find "${DATASET_PATH}" -maxdepth 1 -mindepth 1 -type d | while IFS= read -r CLASS_DIR; do
    CLASS_NAME=$(basename "${CLASS_DIR}")

    # Skip Train and Test directories
    if [[ "$CLASS_NAME" == "Train" || "$CLASS_NAME" == "Test" ]]; then
        continue
    fi

    # Create corresponding class directories in Train and Test
    mkdir -p "${DATASET_PATH}/Train/${CLASS_NAME}"
    mkdir -p "${DATASET_PATH}/Test/${CLASS_NAME}"

    # Get an array of all jpg files in the class directory
    IFS=$'\n' read -d '' -r -a IMAGES < <(find "${CLASS_DIR}" -name '*.jpg' -print0)

    # Calculate the number of images to put in the Train directory
    TOTAL_IMAGES=${#IMAGES[@]}
    TRAIN_COUNT=$(echo "$TRAIN_SPLIT * $TOTAL_IMAGES / 100" | bc | awk '{print int($1+0.5)}')

    # Move the first TRAIN_COUNT to Train and the rest to Test
    for (( i=0; i<${TOTAL_IMAGES}; i++ )); do
        if [ "$i" -lt "$TRAIN_COUNT" ]; then
            # Move to Train
            mv "${IMAGES[$i]}" "${DATASET_PATH}/Train/${CLASS_NAME}"
        else
            # Move to Test
            mv "${IMAGES[$i]}" "${DATASET_PATH}/Test/${CLASS_NAME}"
        fi
    done

    # Delete the original class directory
    rm -rf "${CLASS_DIR}"
done

echo "Dataset splitting completed."
