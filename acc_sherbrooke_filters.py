import os



def mkdirs(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

outpath = './accorderie_sherbrooke'
raw_data_path = './data/raw'
mkdirs(outpath)


snapshot_length = [30, 60,90, 180, 210]

accorderie_id = 109

cmd =  f'python graph_parser/main.py -i={raw_data_path} -o={raw_data_path}/parsed' # output folder will be created if it does not exist
print(cmd)
os.system(cmd)

cmd = f'python graph_filter/main.py -i={raw_data_path}/parsed -o={outpath} --accorderie={accorderie_id}'
print(cmd)
os.system(cmd)



for sn in snapshot_length:
    cmd = f'python graph_metrics/main.py -i={outpath} -o={outpath}/metrics_{sn}/metrics.xlsx -s={sn}'
    print(cmd)
    os.system(cmd)
    #  call labelling
    
    