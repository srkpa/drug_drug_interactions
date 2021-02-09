import json

import click
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["L1000"],
         transformer=["seq"],
         split_mode=["random"],
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
         debug=[False]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[6], batch_size=[32], with_early_stopping=[False])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['conv1d'],
         vocab_size=[151],
         embedding_size=[20],
         cnn_sizes=[
             [300]
         ],
         kernel_size=[[17]],
         pooling_len=[2],
         b_norm=[False],
         pooling=["max"]
         #  use_self_attention=[True]
         )
))

network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    ss_embedding_dim=[32],
    fc_layers_dim=[[]],
    mode=["concat"],
    dropout=[0],
    b_norm=[True],
    is_binary_output=[False],
    op_mode=["cos"]
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
    #is_mtk=[True]
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    optimizer=['adam'],
    weight_decay=[0.0],
    lr=[1e-4],
    loss=['mse'],
    metrics_names=[['imse']],
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
