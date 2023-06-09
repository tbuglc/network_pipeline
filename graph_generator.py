import os
import numpy as np

start_interval = 0.1
end_interval = 1
step = 0.1

root_dir = 'C:\\Users\\bugl2301\\Documents\\generated_graphs'
# Generate graphs

            
for region_bias in np.arange(start=start_interval - 0.1 , stop=end_interval + step, step=step):
    for date_bias in np.arange(start=start_interval, stop=end_interval + step, step=step):
        for sociability_bias in np.arange(start=start_interval, stop=end_interval + step, step=step):
            cmd = f"node graph_generator/src/index.js -o {root_dir}\\r-{0}_s-{sociability_bias}_d-{2}  -u 633 -t 6865 -sd exp -s {sociability_bias} -r {0} -d {2}"
            os.system(cmd)

# valid = [
#     'r-0_s-rand_d-0.1',
#     'r-0_s-rand_d-0.2',
#     'r-0_s-rand_d-0.3',
#     'r-0_s-rand_d-0.4',
#     'r-0_s-rand_d-0.5',
#     'r-0_s-rand_d-0.6',
#     'r-0_s-rand_d-0.7',
#     'r-0_s-rand_d-0.8',
#     'r-0_s-rand_d-0.9',
#     'r-0_s-rand_d-1.0',
# ]

# i =0 
# for walk_dir, sub_dir, files in os.walk(root_dir):
#     if len(sub_dir) == 0 and :
#         i = i + 1
#         print(walk_dir)
#         print(os.stat(walk_dir+'\\metrics.xlsx').st_size)
        
#     # print(walk_dir)
# print(i)

        
# print('Done!')

# Compute metrics for the generated graphs

i =0 
for walk_dir, sub_dir, files in os.walk(root_dir):    
    if len(sub_dir) == 0 and ('metrics.xlsx' not in files or os.stat(walk_dir+'\\metrics.xlsx').st_size < 1024):
        print('Calculating metrics of '+ str(i + 1)+'/122\n')
        
        i = i + 1
        cmd = f'python graph_metrics/main.py -i={walk_dir} -o={walk_dir}\\metrics.xlsx -s={180}'
        print(cmd)
        os.system(cmd)
    # print(walk_dir)
print(i)
print('Done!')



# File size 

# i =0 



# for walk_dir, sub_dir, files in os.walk(root_dir):
    
#     if len(sub_dir) == 0 and 'metrics.xlsx' in files and os.stat(walk_dir+'\\metrics.xlsx').st_size < 1024:
#         i = i + 1
#         # print(walk_dir)
#         # print(os.stat(walk_dir+'\\metrics.xlsx').st_size)
        
#         # 1. read global metrics 
#         # 2. parse folder name 
#         # 3. create an entry with global metrics as features and parsed folder name as label

        
#     # print(walk_dir)
# print(i)
# print('Done!')