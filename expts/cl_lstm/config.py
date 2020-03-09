import json

import click
from ivbase.utils.constants.alphabet import SMILES_ALPHABET
from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],
         transformer=["seq"],
         split_mode=["random", "leave_drugs_out"],
         test_size=[0.10],
         valid_size=[0.15],
         seed=[0],  # 10, 21, 33, 42, 55, 64, 101, 350, 505],
         decagon=[False],
         #use_clusters=[True],
         # use_as_filter=['SOC'],
         use_targets=[False],
         use_side_effect=[False],
         use_pharm=[False],
         label=['ml'],
         debug=[False],
         n_folds=[0],
         test_fold=[0],
         syn=[True]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[256], with_early_stopping=[True])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['lstm'],
         vocab_size=[len(SMILES_ALPHABET) + 2],
         embedding_size=[20],
         lstm_hidden_size=[128],
         nb_lstm_layers=[2],
         dropout=[0.0]
         )
))

AUxNet_params = list(ParameterGrid(dict(
    fc_layers_dim=[[128]],
    output_dim=[32]
)))

network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    fc_layers_dim=[[128] * 2],
    mode=['concat'],
    dropout=[0.10],
    b_norm=[True],
    is_binary_output=[False],
    ss_embedding_dim=[32]
)))

loss_params = list(ParameterGrid(dict(
    use_negative_sampling=[False],
    use_fixed_binary_cost=[False],
    use_fixed_label_cost=[False],
    use_binary_cost_per_batch=[False],
    use_label_cost_per_batch=[False],
    neg_rate=[1.],
    use_sampling=[True],
    samp_weight=[1.]
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    optimizer=['adam'],
    lr=[1e-3],
    loss=['bce'],
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
    loss_params=loss_params,
    dataloader=[True],

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