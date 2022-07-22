# Download
python download.py \
    -sy 2021 \
    -ey 2022 \
    -sm 5 \
    -em 5 \
    -o ../reddit_data/

# Process
python process.py \
    -sy 2021 \
    -ey 2022 \
    -sm 5 \
    -em 5 \
    -o ../reddit_data/ \
    --valid_split_percentage 0.0002
