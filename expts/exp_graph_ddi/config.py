import json

import click
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["drugbank"],
         transformer=["seq"],
         split_mode=["random"],  # leave_drugs_out
         test_size=[0.20],
         valid_size=[0.25],
         use_graph=[True],
         seed=[0]  # , 10, 21, 33, 42, 55, 64, 101, 350, 505]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[1], with_early_stopping=[True])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['lstm'],
         vocab_size=[len(SMILES_ALPHABET) + 2],
         embedding_size=[20],
         lstm_hidden_size=[128],
         nb_lstm_layers=[2],
         dropout=[0.0]
         )
))

graph_network_params = list(ParameterGrid(
    dict(kernel_sizes=[[64] * 2],
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
    lr=[1e-3],
    loss=['bce'],
    metrics_names=[None],
    loss_params=loss_params
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
