import csv
import re

# Define the input and output file paths
name = ''
input_file_path = 'checkpoints/' + name + '/loss_log.txt'
output_file_path = 'checkpoints/' + name + '/loss_log.csv'

# Define the regex pattern to extract the data
pattern = re.compile(r"\(epoch: (\d+), iters: (\d+), time: ([\d.]+), data: ([\d.]+)\)\s*"
                     r"D_A: ([\d.]+) G_A: ([\d.]+) cycle_A: ([\d.]+) idt_A: ([\d.]+)\s*"
                     r"D_B: ([\d.]+) G_B: ([\d.]+) cycle_B: ([\d.]+) idt_B: ([\d.]+)")

# Initialize a list to hold the rows of data
data = []

# Read the input file and extract the data
with open(input_file_path, 'r') as file:
    for line in file:
        match = pattern.match(line)
        if match:
            data.append(match.groups())

# Define the CSV header
header = ['epoch', 'iters', 'time', 'data',
          'D_A', 'G_A', 'cycle_A', 'idt_A',
          'D_B', 'G_B', 'cycle_B', 'idt_B']

# Write the data to a CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerows(data)

print(f"Data successfully written to {output_file_path}")