import os


## parser ##

# input folder: raw csv file
# transactions.csv and members.csv of the global network will be created in output folder 
cmd =  'graph_parser/main.py -i=folder/path -o=folder/path' # output folder will be created if it does not exist
os.system(cmd)

## filter ##

# input folder: global network or filtered network csv's and should contain transaction.csv and members.csv
# output folder: transactions.csv and members.csv of the filtered network will be generated  
cmd = 'graph_filter/main.py -i=folder/path -o=folder/path --accorderie=2'
os.system(cmd)

## metrics ##

# input folder: should contain transactions.csv and members.csv, be it of the global network or filtered
# output file: generated file as in -o argument
cmd = 'graph_metrics/main.py -i=folder/path -o=folder/path/file_name.xlsx'
os.system(cmd)

## report filter ##

# input file: metrics computed from step above\
# output file: generated file with average snapshots
cmd = 'graph_filter/report_filter.py -i=folder/path/file.xlsx -o=folder/path/file_name.xlsx --revenu=30001-50000 --age=55-65 --age=31-54'
os.system(cmd)

## plots ##
# input folder: folder containing metric files. Each file should contain a sheet named "Snapshot Average Metrics"   
cmd = 'graph_plot/main.py -i=folder/path -o=folder/path/file_name.pdf'
os.system(cmd)