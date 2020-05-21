import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid([
    dict(dataset_name=["twosides"],
         transformer=["dgl"],
         split_mode=['leave_drugs_out', "random"],
         test_size=[0.10],
         valid_size=[0.15],
         seed=[42],  # , 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         use_clusters=[False],
         use_as_filter=[None],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False],
         n_folds=[0],
         test_fold=[0],  # 1, 2, 3, 4, 5, 6, 7, 8, 9],
         label=['ml'],
         debug=[True],
         randomized_smiles=[1],
         add=[False]
         ),

dict(dataset_name=["drugbank"],
         transformer=["dgl"],
         split_mode=['leave_drugs_out', "random"],
         test_size=[0.20],
         valid_size=[0.25],
         seed=[42],  # , 10, 21, 33, 42, 55, 64, 101, 350, 505],
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
         randomized_smiles=[100],
         add=[False]
         )
    ]
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[1], batch_size=[32], with_early_stopping=[True])))

pool_arch = list(ParameterGrid(dict(
    arch=[None])))

drug_features_extractor_params = list(ParameterGrid(
    dict(activation=["ReLU"],
         arch=["dglgraph"],
         conv_layer_dims=[[
             [
                 64
             ],
             [
                 64
             ]
         ]],
         dropout=[0.0],
         glayer=["dgl-gin"],
         input_dim=[79],
         pool_arch=pool_arch,
         pooling=["avg"]
         )
))

network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    ss_embedding_dim=[None],
    fc_layers_dim=[[128, 128]],
    mode=["concat"],
    dropout=[0.1],
    b_norm=[True],
    is_binary_output=[False]
)))

loss_params = list(ParameterGrid(dict(
    use_negative_sampling=[False],
    use_fixed_binary_cost=[False],
    use_fixed_label_cost=[False],
    use_binary_cost_per_batch=[False],
    use_label_cost_per_batch=[False],
    neg_rate=[1.],
    use_sampling=[False],
    samp_weight=[False],
    rescale_freq=[False],
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    optimizer=['adam'],
    weight_decay=[0.0],
    lr=[1e-4],
    loss=['bce'],
    metrics_names=[['micro_roc', "macro_roc", "micro_auprc", "macro_auprc"]],
    loss_params=loss_params,
    dataloader=[False]
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
