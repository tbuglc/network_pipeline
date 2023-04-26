import os
import numpy as np

start_interval = 0.1
end_interval = 1
step = 0.1



for region_bias in np.arange(start=start_interval, stop=end_interval + step, step=step):
    for date_bias in np.arange(start=start_interval, stop=end_interval + step, step=step):
        for sociability_bias in np.arange(start=start_interval, stop=end_interval + step, step=step):
            cmd = f"node graph_generator/src/index.js -o C:\\Users\\bugl2301\\Documents\\generated_graphs\\r-{region_bias}_s-{sociability_bias}_d-{date_bias}  -u 633 -t 6865 -sd exp -sp {sociability_bias} -r {region_bias} -d {date_bias}"
            os.system(cmd)
            
            
print('Done!')