import os

accorderie_id = 109
snapshot_length = 180

groups = {}

groups["ages"] = { "18-30" : "--age=18-30" , 
                   "31-55" : "--age=31-54", 
                   "55-65" : "--age=55-65", 
                   "66-80" : "--age=66-80", 
                   "81-95" : "--age=81-95" }
                   #note: did not include 96+, not enough data

groups["revenus"] = { "0" : "--revenu=0", 
                      "0-10000" : "--revenu=0-10000", 
                      "10001-20000" : "--revenu=10001-20000", 
                      "20001-30000" : "--revenu=20001-30000", 
                      "30001-50000" : "--revenu=30001-50000", 
                      "50001-plus" : "--revenu=50001-" }
                      
groups["genres"] = { "0" : "--genre=0", "1" : "--genre=1" }


def mkdirs(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)


## parser ##

# input folder: raw csv file
# transactions.csv and members.csv of the global network will be created in output folder 

outpath_raw = './data'
mkdirs(outpath_raw)

cmd =  f'python graph_parser/main.py -i=./data/raw -o={outpath_raw}' # output folder will be created if it does not exist
print(cmd)
os.system(cmd)

## filter ##

outpath_acc = f'./data/acc{accorderie_id}'
mkdirs(outpath_acc)

# input folder: global network or filtered network csv's and should contain transaction.csv and members.csv
# output folder: transactions.csv and members.csv of the filtered network will be generated  
cmd = f'python graph_filter/main.py -i={outpath_raw} -o={outpath_acc} --accorderie={accorderie_id}'
print(cmd)
os.system(cmd)

# Date filter
#cmd = "graph_filter/main.py -i=folder/path -o=folder/path --date ':06/2010,10/2016'" # date between 06/2010 and 10/2016
#cmd = "graph_filter/main.py -i=folder/path -o=folder/path --date '>06/2010'" # date after 06/2010 
#cmd = "graph_filter/main.py -i=folder/path -o=folder/path --date '<06/2006'" # date before 06/2006
#os.system(cmd)

## metrics ##

outpath_xlsx = f'./data/acc{accorderie_id}/data.xlsx'
mkdirs(os.path.dirname(outpath_xlsx))

# input folder: should contain transactions.csv and members.csv, be it of the global network or filtered
# output file: generated file as in -o argument
cmd = f'python graph_metrics/main.py -i={outpath_acc} -o={outpath_xlsx} -s={snapshot_length}'
print(cmd)
os.system(cmd)

## report filter ##


for group_name in groups:
    #here, group_name is e.g. "age", "revenu", "genre"
    #groups[group_name] is a dict, with key = filter_name, val = filter_str
    for filter_name in groups[group_name]: 
        filter_str = groups[group_name][filter_name]
         
        outpath_filtered_xlsx = f'./data/acc{accorderie_id}/{group_name}/data_{filter_name}.xlsx'
        mkdirs(os.path.dirname(outpath_filtered_xlsx))
        
        cmd = f'python graph_filter/report_filter.py -i={outpath_xlsx} -o={outpath_filtered_xlsx} {filter_str}'
        print(cmd)
        os.system(cmd)


    ## plots ##

    inpath_plots = f'./data/acc{accorderie_id}/{group_name}'
    outpath_plots = f'./data/acc{accorderie_id}/{group_name}/pdf/data.pdf'
    mkdirs(os.path.dirname(outpath_plots))

    # input folder: folder containing metric files. Each file should contain a sheet named "Snapshot Average Metrics"   
    cmd = f'python graph_plot/main.py -i={inpath_plots} -o={outpath_plots}'
    print(cmd)
    os.system(cmd)

    # filter by date 
    # -i={outpath_raw} -o={outpath_acc} --accorderie={accorderie_id}
    cmd = "graph_filter/main.py -i=folder/path -o=folder/path --date '>10/2012'"
    print(cmd)
    os.system(cmd)






