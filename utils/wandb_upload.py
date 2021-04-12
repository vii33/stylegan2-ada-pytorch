from typing import Dict, List
from pathlib import Path
import click
import os
import json
from collections import defaultdict

import wandb
#os.environ['WANDB_MODE'] = 'dryrun'


def extract_kimg_from_metric_dict(pkl_dict:Dict) -> str:
    '''
    Gets the kimg number for the pkl file. Example format: network-snapshot-000320.pkl
    '''

    if "snapshot_pkl" in pkl_dict:
        name = pkl_dict['snapshot_pkl'].split(".")[-2]
        return name[-6:] 


def load_metric_jsonls(folder_path: str, metric_type: str) -> list:
    '''
    Finds all .jsonl files in a given folder and loads their json entries in a dict.
    '''
    parent = Path(folder_path)

    filepaths = parent.rglob(f"metric-{metric_type}*.jsonl")

    mydict = {}
    for path in filepaths:
        title = path.stem.split("-")[1].split("-")[0]
        mydict[title] = path
    
    return mydict

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value == '':
            return []
        return value.split(',')

@click.command()

@click.option('--metrics', help='Comma-separated list of metrics [fid50k_full, kid50k_full]', required=True, type=CommaSeparatedList())
@click.option('--metrics_path', help='Folder where metrics can be found', required=True, type=click.STRING)
@click.option('--wandb_api_key', help='WandB API Key for login', required=True, type=click.STRING)
@click.option('--wandb_group_id', help='WandB Group ID (same as training)', required=True, type=click.STRING)

def main(metrics:list, metrics_path:str, wandb_api_key:str, wandb_group_id:str):
    os.environ["WANDB_API_KEY"] = wandb_api_key

    met_list = []
    for met_type in metrics:
        met_paths = load_metric_jsonls(metrics_path, met_type)

        mets = {}
        for met_key, met_path in met_paths.items():
            with open(met_path, 'r') as json_file:
                met_list.extend( list(json_file) )
    

    kimgs = 0
    met_dict = defaultdict(dict)
    for item in met_list:
        met_con = json.loads(item)
        kimgs = int ( extract_kimg_from_metric_dict(met_con) ) 

        met_dict[kimgs][ met_con['metric'] ] =   next(iter( met_con['results'].values() ))  # get first (and only) val from results key (is a subdict)


    run = wandb.init(
        project = "stylegan2ada-py",
        group = wandb_group_id,
        job_type="training_val",
        config = { 'pkl path' : metrics_path } 
        )

    for key, value_dict in met_dict.items():
        print(f"Logging metric: at step: {int(key)}, value: {value_dict}")
        value_dict['kimg'] = int(key)
        wandb.log( value_dict )
    run.finish()




#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() 

#----------------------------------------------------------------------------