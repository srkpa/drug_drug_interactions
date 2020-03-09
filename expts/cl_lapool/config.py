import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["adj"],
         split_mode=["leave_drugs_out", "random"],
         test_size=[0.10],
         valid_size=[0.15],
         seed=[0, 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         use_clusters=[True],
         use_as_filter=['SOC'],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False],
         label=['ml'],
         debug=[False],
         n_folds=[0],
         test_fold=[0]

         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[256], with_early_stopping=[True])))

pool_arch = list(ParameterGrid(
    dict(arch=["laplacian"],
         hop=[3],
         reg_mode=[1],
         lap_hop=[1],
         attn=[1],
         concat=[True]
         )
))
drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['dglgraph'],
         input_dim=[79],
         conv_layer_dims=[[[64], [64]]],
         dropout=[0.],
         gather=["attn"],
         gather_dim=[64],
         pool_arch=pool_arch,
         activation=['ReLU'],
         glayer=["th-gin"]
         )
))

network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    fc_layers_dim=[[128] * 2],
    mode=['concat'],
    att_mode=[None],
    dropout=[0.10],
    b_norm=[True],
    is_binary_output=[False],
    ss_embedding_dim=[None]
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
    lr=[1e-3],
    loss=['bce'],
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
    loss_params=loss_params,
    dataloader=[False],
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