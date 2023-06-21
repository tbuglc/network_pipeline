import os
import numpy as np

start_interval = 0.1
end_interval = 1
step = 0.1

root_dir = 'C:\\Users\\bugl2301\\Documents\\generated_graphs'
# Generate graphs


region_range = np.arange(start=start_interval - step , stop=end_interval + step, step=step)
date_range = np.arange(start=start_interval - step , stop= 2 + step, step=step)
sociability_range = np.arange(start=start_interval , stop=end_interval + step, step=step)
sociability_distribution_range = ['rand','exp']

count = 1
iterations = np.arange(9, 51, 1)
for i in iterations:
    for sd in sociability_distribution_range:
        for region_bias in region_range:
            for date_bias in date_range:
                for sociability_bias in sociability_range:
                    count += 1
                    print(f'count-{count}-it={i}-sd={sd}-region={region_bias}-date={date_bias}-sociability={sociability_bias}')
                    cmd = f"node graph_generator/src/index.js -o {root_dir}\\iter-{i}_r-{region_bias}_sd-{sd}_sp-{sociability_bias}_d-{date_bias}  -u 592 -t 6754 -sd {sd} -sp {sociability_bias} -r {region_bias} -d {date_bias}"
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

# i =0 
# for walk_dir, sub_dir, files in os.walk(root_dir):    
#     if len(sub_dir) == 0 and ('metrics.xlsx' not in files or os.stat(walk_dir+'\\metrics.xlsx').st_size < 1024):
#         print('Calculating metrics of '+ str(i + 1)+'/122\n')
        
#         i = i + 1
#         cmd = f'python graph_metrics/main.py -i={walk_dir} -o={walk_dir}\\metrics.xlsx -s={180}'
#         print(cmd)
#         os.system(cmd)
#     # print(walk_dir)
# print(i)
# print('Done!')



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