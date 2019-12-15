import json

import click
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["seq"],
         split_mode=["leave_drugs_out"],
         test_size=[0.15],
         valid_size=[0.10],
         use_graph=[True],
         seed=[42]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[256], with_early_stopping=[True])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['conv1d'],
         vocab_size=[len(SMILES_ALPHABET) + 2],
         embedding_size=[20],
         cnn_sizes=[
             [256] * 4, [512] * 4
         ],
         kernel_size=[[5], [10]],
         dilatation_rate=[1],
         pooling_len=[2],
         b_norm=[False])
))

graph_network_params = list(ParameterGrid(
    dict(kernel_sizes=[[64] * 2, [64] * 4],
         activation=['relu'],
         b_norm=[False])
))

network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    graph_network_params=graph_network_params,
    edges_embedding_dim=[64],
    tied_weights=[True],
    fc_layers_dim=[[128] * 2],
    mode=["concat"],
    att_hidden_dim=[None],
    dropout=[0.25],
    b_norm=[False],
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
    lr=[1e-4],
    loss=['bce'],
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
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
