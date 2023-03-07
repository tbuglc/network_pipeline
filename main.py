import os
from InquirerPy import prompt, inquirer
import subprocess as sb
from InquirerPy.validator import NumberValidator

os.environ['PYTHONPATH'] = os.path.abspath('.')

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
            sb.run(['python', 'graph_metrics/main.py', '--span=' + str(span), '--folder_name=' + str(flder)])
    if entry_query == 4:
        accorderies = inquirer.checkbox(
            message="Select Accorderie(s)",
            choices=os.listdir('data/metrics'),
            validate=lambda r: len(r) >= 1,
            invalid_message="should be at least 1 selection",
            instruction="(select at least 1)",
        ).execute()

        query = inquirer.text(
            message="Filter query (i.e --age=34-55 --age=23-43 --ville=sherbrooke): ",
        ).execute()

        for acc in accorderies:
            command = ['python', 'graph_filter/report_filter.py', '--folder_name=' + acc]

            for q in query.split(' '):
                command.append(q)

            sb.run(command)

    if entry_query == 5:
        fils = []
        for fd in os.listdir('data/metrics'):
            print(fd)
            fl_m = inquirer.checkbox(
                message="Select metric file in "+fd,
                choices=os.listdir('data/metrics/' + fd),
            ).execute()

            fils.append(fl_m)
        print(fils)
        # sb.run(['python', 'graph_plot/main.py'])
