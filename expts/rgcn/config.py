import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["deepddi"],
         split_mode=["random"],  # "leave_drugs_out"
         test_size=[0.10],
         valid_size=[0.15],
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
    dict(evaluate_every=[5], n_epochs=[10], eval_batch_size=[5], with_early_stopping=[True])))

network_params = list(ParameterGrid(dict(
    network_name=['RGCN'],
    h_dim=[500],
    num_bases=[100],
    num_hidden_layers=[2],
    dropout=[0.2],
    reg_param=[0.01],
    n_bases=[100]
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
    optimizer=['adam'],
    lr=[1e-2],
    loss=['bce'],
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
    graph_batch_size=[30000],
    graph_split_size=[0.5],
    edge_sampler=["uniform"]
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
