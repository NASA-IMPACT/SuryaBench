#!/bin/bash

# Parameters
fulldisk_list="./SuryaBench/scripts/fulldisklist.txt"  # Path to the list of HARP numbers
EXECUTABLE_PATH="./SuryaBench/build/SuryaBench"  # Path to the executable
PROGRESS_BAR_LENGTH=50  # Length of the progress bar

# Function to draw a progress bar
# Arguments: current_iteration max_iterations bar_length current_harp_number
draw_progress_bar() {
    local current=$1
    local total=$2
    local length=$3
    local harp_num=$4
    local percent=$((100 * current / total))
    local filled=$((length * current / total))
    local empty=$((length - filled))

    # Print the progress bar
    printf "\r["
    printf "%0.s#" $(seq 1 $filled)
    printf "%0.s " $(seq 1 $empty)
    printf "] %s%% (HARP: %s)" "$percent" "$harp_num"
}

# Count total HARP numbers
total=$(wc -l < "$fulldisk_list")
current=0

# Read the text file line by line
while IFS= read -r harp_number; do
    ((current++))

    "$EXECUTABLE_PATH" "$harp_number"

    # Draw the progress bar after each iteration
    draw_progress_bar $current $total $PROGRESS_BAR_LENGTH $harp_number

done < "$fulldisk_list"

# Newline to finish
echo ""
echo "Computations finished!"
