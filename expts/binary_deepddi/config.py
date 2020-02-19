import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["deepddi"],
         split_mode=["leave_drugs_out", "random"],  #
         test_size=[0.10],
         valid_size=[0.15],
         seed=[0, 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         use_clusters=[False],
         use_as_filter=[None],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False],
         n_folds=[0],
         test_fold=[0],  # 1, 2, 3, 4, 5, 6, 7, 8, 9],
         label=['ml'],
         debug=[False],
         density=[True]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[2], batch_size=[256], with_early_stopping=[True])))

network_params = list(ParameterGrid(dict(
    network_name=['deepddi'],
    input_dim=[100],
    hidden_sizes=[[2048] * 5],
    is_binary_output=[True],
    ss_embedding_dim=[32]
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
    lr=[1e-5],
    loss=['bce'], #[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']
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
