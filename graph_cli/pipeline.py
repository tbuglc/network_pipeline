import os

from InquirerPy import prompt, inquirer
import subprocess as sb
from InquirerPy.validator import NumberValidator

questions = [
    {"type": "input", "message": "\n1) Parse \n2) Split Accorderie \n3) Metrics \n4) Filters \n5) Plots \n0) Exit \n",
     "name": "entry_query"},
]

accorderies = [2, 86, 88, 109, 104, 114, 92, 115, 108, 111, 110, 117, 112, 119, 118, 113, 120, 116, 121]

is_on = True
while is_on:
    result = prompt(questions)

    entry_query = int(result['entry_query'])

    if entry_query == 0:
        is_on = False

    if entry_query == 1:
        sb.run(['python', 'graph_parser/main.py'])

    if entry_query == 2:
        for acc in accorderies:
            sb.run(['python', 'graph_filter/main.py', '--accorderie=' + str(acc)])

    if entry_query == 3:
        span = inquirer.text(
            message="Snapshot time window:",
            validate=NumberValidator(),
            default="30",
            filter=lambda r: int(r),
        ).execute()

        acc_folders = inquirer.checkbox(
            message="Select Accorderie(s)",
            choices=os.listdir('data/accorderies'),
            validate=lambda r: len(r) >= 1,
            invalid_message="should be at least 1 selection",
            instruction="(select at least 1)",
        ).execute()

        for flder in acc_folders:
            sb.run(['python', 'graph_analysis/main.py', '--span=' + str(span), '--folder_name=' + str(flder)])

# show options
# work on structuring options select,
# restructure data and output
