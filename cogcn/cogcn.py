from train import train
import os
import sys
import json
import argparse
import numpy as np
import warnings
from pathlib import Path
from time import time
import pickle
from tabulate import tabulate
from hyperopt import hp, fmin, tpe


warnings.filterwarnings('ignore')

# Add project source to path
root = Path(os.path.abspath(os.path.join(
    os.getcwd().split('Modified-CoGCN')[0], 'Modified-CoGCN')))

if root not in sys.path:
    sys.path.append(root.__str__())

from metrics.metrics_util import *

# If individually weighted is desired
weights = {
    'jpetstore': {
        'ifn': 1.18,
        'sm': 1.45
    },
    'plants': {
        'ifn': 1.3,
        'sm': 1.67
    },
    'acmeair': {
        'ifn': 1.31,
        'sm': 1.23
    },
    'daytrader': {
        'ifn': 1.33,
        'mq': 1.86
    }
}

def get_entropy(x):
    probs = x / sum(x)
    return -sum(probs * np.log(probs))

class obj(object):
    def __init__(self, d: dict):
        self.__dict__ = d


def get_args(choices):
    d = {
        'model': 'gcn_vae',
        'seed': 42,
        'preepochs': 350,
        'clusepochs': 1,
        'epochs': 300,
        'lr': 0.01,
        'outfile': 'embeddings',
        'verbose': False,
        'dumplogs': False
    }

    hp_space_dict = {}

    for key, value in choices.items():
        if isinstance(value, tuple):
            if isinstance(value[0], int):
                hp_space_dict[key] = hp.randint(
                    key, value[1] - value[0]) + value[0]
                d[key] = np.random.randint(value[0], value[1] + 1)
            else:
                hp_space_dict[key] = hp.uniform(key, value[0], value[1])
                d[key] = np.random.uniform(value[0], value[1])
        else:
            hp_space_dict[key] = hp.choice(key, value)
            d[key] = np.random.choice(value)

    hp_space = hp.choice('params', [hp_space_dict])
    return d, hp_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # 5000 epochs is where the pretrain loss starts hovering around 5.0 - 6.0
    parser.add_argument('--k', type=int, default=6, help='Number of clusters.')
    parser.add_argument('--preepochs', type=int, default=350,
                        help='Number of epochs to pre-train.')
    parser.add_argument('--clusepochs', type=int, default=1,
                        help='Number of epochs to pre-train for clustering.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--outfile', type=str,
                        default='embeddings', help='output embeddings file.')
    parser.add_argument('--dumplogs', type=bool,
                        default=False, help='Print logs to file.')

    # Override the above for hyper-parameter tuning
    argchoice = {
        'hidden1': (8, 64),
        'hidden2': (4, 32),
        'lambda1': (0., 1.),
        'lambda2': (0., 1.),
        'lambda3': (0., 1.),
        'dropout': (0.2, 0.5),
        'verbose': [False]
    }

    args = parser.parse_args()

    # !! CLIENT DATA !! #
    partition_sizes = {
        'acmeair': range(3, 13, 2),
        'daytrader': range(3, 36, 2),
        'jpetstore': range(3, 18, 2),
        'plants': range(2, 13, 2)
    }
    # !! CLIENT DATA !! #

    for proj_path in root.joinpath('datasets_runtime').glob("*"):
        if not proj_path.is_dir():
            continue

        print("###", proj_path.stem.upper(), "\n")
        results_table = []
        header = ["Partitions", "BCS[-]", "ICP[-]",
                  "SM[+]", "MQ[+]", "IFN[-]", "NED[-]", "Entropy[+]", "WC_time[-]", "Loss[-]"]
        cogcn_input = proj_path.joinpath('cogcn_input')
        cogcn_output = proj_path.joinpath('cogcn_output')

        # Some files required to compute metrics
        with open(proj_path.joinpath("mono2micro_output", "bcs_per_class.json"), 'r') as f:
            bcs_per_class = json.load(f)

        with open(proj_path.joinpath("mono2micro_output", "runtime_call_volume.json"), 'r') as f:
            runtime_call_volume = json.load(f)

        cluster_size = list(partition_sizes[proj_path.stem])
        args, hp_space = get_args(argchoice)

        def objective(hp_args):
            all_args = obj({**args, **hp_args})
            # Run CoGCN
            retries = 0
            while retries < 3:
                now = time()
                k = np.random.randint(cluster_size[0], cluster_size[1] + 1)

                try:
                    partitions = train(all_args, num_clusters=k, datapath=cogcn_input)
                    break
                except:
                    print(f'Iteration failed; retrying {retries} / 3')
                    retries += 1
            
            if retries == 3:
                return 100

            # Get the mapping file
            with open(cogcn_output.joinpath('mapping.json')) as mapping_file:
                mapping = json.load(mapping_file)

            try:
                # Save vertical partition assignments
                vertical_cluster_assignment = {class_name: int(
                    partitions[id_]) for class_name, id_ in mapping.items()}
            except err:
                print('[ERR] Could not save cluster assignment. Check that ' +\
                        f'mapping.json is in datasets_runtime/{proj_path.stem}/cogcn_output' +\
                        ' directory and has the entries of the form "ClassName": int in the ' +\
                        'json.')
                print(err)

            with open(cogcn_output.joinpath('vertical_cluster_assignment__{}.json'.format(k)), 'w') as cluster_assgn_file:
                json.dump(vertical_cluster_assignment,
                          cluster_assgn_file, indent=4)

            # Compute Metrics
            ROOT = 'Root'
            class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(
                vertical_cluster_assignment, bcs_per_class)
            bcs = business_context_purity(partition_class_bcs_assignment)
            icp = inter_call_percentage(
                ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = structural_modularity(
                partition_class_bcs_assignment, runtime_call_volume)
            mq = modular_quality(
                ROOT, partition_class_bcs_assignment, runtime_call_volume)
            ifn = interface_number(
                ROOT, partition_class_bcs_assignment, runtime_call_volume)
            ned, class_sizes = non_extreme_distribution(partition_class_bcs_assignment)
            entropy = get_entropy(class_sizes)

            # Loss function with weights learned from data.
            loss = - weights[proj_path.stem].get('sm', 1.) * sm + \
                weights[proj_path.stem].get('icp', 1.) * icp + \
                weights[proj_path.stem].get('bcs', 1.) * bcs / 10.

            cur_result = [[k, str(bcs), str(icp), str(
                sm), str(mq), str(ifn), str(ned), str(entropy), round(time()-now, 2), loss]]
            results_table.append(cur_result[0])
            #print(tabulate(cur_result, header, tablefmt='tsv'))

            return loss

        best = fmin(objective, hp_space, algo=tpe.suggest, max_evals=100)
        print(best)
        print(tabulate(results_table, header, tablefmt="tsv"))
        print("\n")


if __name__ == '__main__':
    main()
