## Installation

1. Graph Parser 

```sh
$ cd graph_parser
graph_parser $ python -m venv venv
graph_parser $ .\venv\Scripts\Activate.ps1
graph_parser $ pip install -r requirements.txt
 
```

2. Graph Filter 

```sh
$ cd graph_filter
graph_filter $ python -m venv venv
graph_filter $ .\venv\Scripts\Activate.ps1
graph_filter $ pip install -r requirements.txt
 
```

3. Graph Metrics 

```sh
$ cd graph_metrics
graph_metrics $ python -m venv venv
graph_metrics $ .\venv\Scripts\Activate.ps1
graph_metrics $ pip install -r requirements.txt
```


## Execution

1. Graph Parser 

```sh
$ cd graph_parser
# csv files are expected to be in input folder
graph_parser $ python .\main.py --input=path --output=path # this command will generate members.csv and transactions.csv in 
```


2. Graph Filter: Split accorderies or Filter metrics report

```sh
$ cd graph_filter
# csv files are expected to be in input folder
graph_filter $ python .\accorderie_filter.py --input=path --output=path --accorderie=2 # this command will filter members and transactions in accorderie 2 AND age between 18-30 or 23-34

graph_filter $ python .\report_filter.py --input=path --output=path --age=18-30 --age=55-65 --folder_name=experiment_101# this command will filter members and transactions in accorderie 2 AND age  between 18-30 OR 55-66
```

3. Graph Metrics 

```sh
$ cd graph_metrics
# csv files are expected to be in input folder
graph_metrics $ python .\main.py --input=path --output=path --span=30 --folder_name=experiment_101 # default span=30 and folder_name=Metrics
```
4. Graph Plot 

```sh
$ cd graph_plot
# csv files are expected to be in input folder
graph_plot $ python .\main.py --input=path --output=path --folder_name=experiment_101
```