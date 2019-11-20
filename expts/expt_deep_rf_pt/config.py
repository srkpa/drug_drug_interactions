import click
import json
from sklearn.model_selection import ParameterGrid
import os

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["seq"],
         split_mode=["random"],  # "leave_drugs_out"
         test_size=[0.10],
         valid_size=[0.15],
         seed=[0],  # , 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         use_clusters=[False],
         use_as_filter=[None],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False]
         )
))

pretrained_drug_features_extractor_params = list(ParameterGrid(
    dict(directory=[
        "{}/datasets-ressources/DDI/twosides/pretrained_models/random/drugbank".format(
            os.environ["INVIVO_CACHE_ROOT"])],
        delete_layers=["last"],
        output_dim=[86]
    )
))

network_params = list(ParameterGrid(dict(
    network_name=['deeprf'],
    n_estimators=[100],
    random_state=[42],
    n_jobs=[-1],
    class_weight=[None],  # "balanced", "balanced_subsample"],
    nb_chains=[None],
    pretrained_drug_features_extractor_params=pretrained_drug_features_extractor_params,
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
    option=["default"]
)))


@click.command()
@click.option('-d', '--dataset', type=str, default='twosides', help='folder to scan')
@click.option('-a', '--algo', type=str, default='bmnddi')
@click.option('-o', '--outfile', type=str, default='configs.json')
def generate_config_file(dataset, algo, outfile):
    # todo: use argument of the function

    expt_config = dict(
        model_params=model_params,
        dataset_params=dataset_params,
    )

    with open(outfile, 'w') as f:
        json.dump(expt_config, f, indent=2)


if __name__ == '__main__':
    generate_config_file()
