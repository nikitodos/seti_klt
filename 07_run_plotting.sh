#!/bin/bash

# Example of running
# ./07_run_plotting.sh /home/obs/klt/experiment_dd-mm-yyyy "600,10000000,7158279,357914,71583"
# this will plot all .fil files from directory of your experiment results in ranges of 600 channels, 10 000 000 channels, etc

unset PYTHONPATH
source /opt/conda/bin/activate klt
python --version

# Check if the user provided a directory as an argument
if [ $# -ne 2 ]; then
  echo "Usage: $0 <directory> <list_of_values>"
  exit 1
fi

# Get the directory path and list of values from the command-line arguments
directory="$1"
values="$2"

# Check if the specified directory exists
if [ ! -d "$directory" ]; then
  echo "Error: The specified directory does not exist."
  exit 1
fi

# Split the list of values into an array
IFS=',' read -ra value_array <<< "$values"

# Loop through files in the directory ending with .fil and run the Python script
for file in "$directory"/*.fil; do
  if [ -f "$file" ]; then
    for value in "${value_array[@]}"; do
      # Extract the file name without the extension
      file_name_without_extension=$(basename -- "$file")
      file_name_without_extension="${file_name_without_extension%.*}"
      # Define the output name by combining the file name and the value
      output_name="${file_name_without_extension}_c${value}"
      echo "Running 08_plot_data.py for file: $file with value: $value"
      python3 08_plot_data.py -f "$file" -s -c 0 "$value" -n "$output_name"
    done
  fi
done

echo "Script completed."