#!/bin/bash

# === Config ===
BUCKET="pcse555storage"
TRAINFOLDER="train"
TESTFOLDER="test"
LOCAL_BASE_DIR="data"
TRAIN_DIR="$LOCAL_BASE_DIR/train/"
TEST_DIR="$LOCAL_BASE_DIR/test/"

# === User Input ===
read -p "Download train, test, or both (train,test,both)? " SET
read -p "How many images? (enter number) " COUNT

# === Function to Download Images ===
download() {
    local local_dir=$1     # e.g., data/train/
    local s3_folder=$2     # e.g., train/

    mkdir -p "$local_dir"

    echo "Downloading $COUNT images from s3://$BUCKET/$s3_folder to $local_dir"

    aws s3 ls "s3://$BUCKET/$s3_folder/gt_post/" | head -n "$COUNT" | awk '{print $4}' | while read -r file; do
        if [ -n "$file" ]; then
            echo " -> Downloading: $file"
            aws s3 cp "s3://$BUCKET/$s3_folder/gt_post/$file" "$local_dir"
        fi
    done
}

# === Main Logic ===
case "$SET" in
  train)
    download "$TRAIN_DIR" "$TRAINFOLDER"
    ;;
  test)
    download "$TEST_DIR" "$TESTFOLDER"
    ;;
  both)
    download "$TRAIN_DIR" "$TRAINFOLDER"
    download "$TEST_DIR" "$TESTFOLDER"
    ;;
  *)
    echo "❌ '$SET' is not a valid option. Please choose: train, test, or both."
    ;;
esac
