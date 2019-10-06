import click
import json
from sklearn.model_selection import ParameterGrid
from ivbase.utils.constants.alphabet import SMILES_ALPHABET

dataset_params = list(ParameterGrid(
    dict(dataset_name=["twosides"],  # "twosides_icd11" drugfood
         transformer=["seq"],
         split_mode=["random"], #"leave_drugs_out"
         test_size=[0.15],
         valid_size=[0.10],
         seed=[42], #0, 10, 21, 33, 42, 55, 64, 101, 350, 505
         decagon=[False],
         use_side_effects_mapping=[False]
         #use_as_filter=["PT"]
         )
))

fit_params = list(ParameterGrid(
    dict(n_epochs=[100], batch_size=[256], with_early_stopping=[False])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['conv1d'],
         vocab_size=[len(SMILES_ALPHABET) + 2],
         embedding_size=[20],
         cnn_sizes=[
             [128] * 4
         ],
         kernel_size=[[17]],
         dilatation_rate=[1],
         #pooling=["max"],
         pooling_len=[2],
         b_norm=[False],
         normalize_features=[True],
         use_self_attention=[False])
))

# network_params = list(ParameterGrid(dict(
#     network_name=['bmnddi'],
#     drug_feature_extractor_params=drug_features_extractor_params,
#     fc_layers_dim=[[128]*4],
#     mode= ["concat"],
#     dropout=[0],
#     b_norm=[True],
# )))


network_params = list(ParameterGrid(dict(
    network_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    fc_layers_dim=[[128] * 2],
    mode=['concat'],
    att_hidden_dim=[32],
    dropout=[0.15],
    b_norm=[True]
)))

model_params = list(ParameterGrid(dict(
    network_params=network_params,
    optimizer=['adam'],
    lr=[1e-3],
    loss=['bce'],
    metrics_names=[['macro_roc', 'macro_auprc', 'micro_roc', 'micro_auprc']],
    use_negative_sampled_loss=[False]
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
