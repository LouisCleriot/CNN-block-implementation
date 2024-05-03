#!/bin/bash
# Usage:
# chmod +x split_dataset.sh
# ./split_dataset.sh /path/to/your/dataset 80

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 path_to_dataset train_split_percentage"
    exit 1
fi

DATASET_PATH="$1"
TRAIN_SPLIT="$2"

if ! [[ "$TRAIN_SPLIT" =~ ^[0-9]+$ ]] || [ "$TRAIN_SPLIT" -lt 0 ] || [ "$TRAIN_SPLIT" -gt 100 ]; then
    echo "The training split percentage must be a number between 0 and 100."
    exit 1
fi

mkdir -p "${DATASET_PATH}/Train"
mkdir -p "${DATASET_PATH}/Test"

find "${DATASET_PATH}" -maxdepth 1 -mindepth 1 -type d | while IFS= read -r CLASS_DIR; do
    CLASS_NAME=$(basename "${CLASS_DIR}")
    if [[ "$CLASS_NAME" == "Train" || "$CLASS_NAME" == "Test" ]]; then
        continue
    fi

    mkdir -p "${DATASET_PATH}/Train/${CLASS_NAME}"
    mkdir -p "${DATASET_PATH}/Test/${CLASS_NAME}"

    # Correcting how images are gathered and processed
    IFS=$'\n' IMAGES=($(find "${CLASS_DIR}" -type f -name '*.jpg'))
    echo "Found ${#IMAGES[@]} images in ${CLASS_DIR}"

    TOTAL_IMAGES=${#IMAGES[@]}
    TRAIN_COUNT=$(echo "$TRAIN_SPLIT * $TOTAL_IMAGES / 100" | bc | awk '{print int($1+0.5)}')

    for (( i=0; i<${TOTAL_IMAGES}; i++ )); do
        if [ "$i" -lt "$TRAIN_COUNT" ]; then
            mv "${IMAGES[$i]}" "${DATASET_PATH}/Train/${CLASS_NAME}"
        else
            mv "${IMAGES[$i]}" "${DATASET_PATH}/Test/${CLASS_NAME}"
        fi
    done

    rm -rf "${CLASS_DIR}"
done

echo "Dataset splitting completed."
