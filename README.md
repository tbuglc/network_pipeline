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

3. Graph Analysis 

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
graph_parser $ python .\main.py # this command will generate members.csv and transactions.csv in 
```


2. Graph Filter 

```sh
$ cd graph_filter
# csv files are expected to be in input folder
graph_filter $ python .\main.py --age=18-30 --accorderie=2 # this command will filter members and transactions in accorderie 2 AND age between 18-30

graph_filter $ python .\main.py --age=18-30 --age=55-65 --accorderie=2 # this command will filter members and transactions in accorderie 2 AND age  between 18-30 OR 55-66
```

3. Graph Analysis 

```sh
$ cd graph_metrics
# csv files are expected to be in input folder
graph_metrics $ python .\main.py --span=30 --folder_name=experiment_101 # default span=30 and folder_name=analysis
```