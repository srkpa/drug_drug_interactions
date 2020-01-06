import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["seq"],
         split_mode=["random"],
         test_size=[0.15],
         valid_size=[0.10],
         seed=[42],
         debug=[False]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[1], batch_size=[256], patience=[8], l2_lambda=[0])))

network_params = list(ParameterGrid(dict(
    network_name=['mhcaddi'],
    d_hid=[32],
    d_atom_feat=[3],
    n_prop_step=[3],
    score_fn=['trans'],
    n_attention_head=[1],
    d_readout=[32],
    transR=[False],
    transH=[False],
    dropout=[0.1],
)))


model_params = list(ParameterGrid(dict(
    network_params=network_params,
    learning_rate=[1e-3],
    train_neg_pos_ratio=[1],
    fold_i=[3]
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
        fit_params=fit_params
    )

    with open(outfile, 'w') as f:
        json.dump(expt_config, f, indent=2)


if __name__ == '__main__':
    generate_config_file()
