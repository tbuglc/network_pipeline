import csv

# Replace 'input_file.csv' with the path to your input CSV file
input_file = 'shared_result_file.csv'
output_data_file = 'data.csv'
output_targets_file = 'targets.csv'

def process_line(line):
    # Convert line to a list of numbers
    return list(map(float, line.strip().split(',')))

with open(input_file, 'r') as infile, open(output_data_file, 'w', newline='') as data_file, open(output_targets_file, 'w', newline='') as targets_file:
    data_writer = csv.writer(data_file)
    targets_writer = csv.writer(targets_file)

    for line in infile:
        # Convert each line into a list of numbers
        line_numbers = process_line(line)

        # Write the last 3 numbers to 'targets' file
        targets_writer.writerow(line_numbers[-3:])

        # Write the remaining numbers to 'data' file
        data_writer.writerow(line_numbers[:-3])

print("Processing completed. Data and Targets CSV files have been created.")
