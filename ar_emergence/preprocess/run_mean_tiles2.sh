#!/bin/bash

# List of unique numbers
unique_numbers=(
11130
11149
11158
11162
11190
11199
)

# Loop through each unique number and run the python script
for number in "${unique_numbers[@]}"; do
  python3 mean_tiles.py $number
done

