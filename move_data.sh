# Define scratch path and dataset source
LOCAL_DATA_PATH="/scratch/local/25506805/ImageNet1k"
SOURCE_DATA_PATH="/blue/weishao/chojnowski.h/data/imagenet-1k"
 
# Make sure local scratch directory exists
mkdir -p "$LOCAL_DATA_PATH"
 
# Copy data to local scratch (use rsync to handle partial/incremental copy)
echo "Copying dataset to local scratch..."
rsync -a --info=progress2 "$SOURCE_DATA_PATH/" "$LOCAL_DATA_PATH/"
 
echo "✅ Dataset copied to local scratch: $LOCAL_DATA_PATH"
 
export LOCAL_DATA_PATH