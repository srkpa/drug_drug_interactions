import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["drugbank"],
         transformer=["lee"],
         split_mode=["random"],  # "leave_drugs_out"
         test_size=[0.20],
         valid_size=[0.25],
         seed=[42],  # , 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         use_clusters=[False],
         use_as_filter=[None],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[256], with_early_stopping=[True])))

network_params = list(ParameterGrid(dict(
    network_name=['adnn'],
    hidden_layer_dims=[[1000]],
    output_dim=[200],
    activation=['relu'],
    dropout=[0],
    b_norm=[False],
    mn_hidden_layer_dims=[[2000] * 6]
)))

loss_params = list(ParameterGrid(dict(
    use_negative_sampling=[False],
    use_fixed_binary_cost=[False],
    use_fixed_label_cost=[False],
    use_binary_cost_per_batch=[False],
    use_label_cost_per_batch=[False]
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    optimizer=['rmsprop'],
    lr=[1e-3],
    loss=['bce'],
    metrics_names=[[]],
    loss_params=loss_params,
    dataloader=[True]
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
