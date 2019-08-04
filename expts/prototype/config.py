import click
import json
from sklearn.model_selection import ParameterGrid
from ivbase.utils.constants.alphabet import SMILES_ALPHABET


dataset_params = [dict(
    name="twosides",
    smi_transf= "seq",
    inv_seed=None,
    seed= 42,
)]

fit_params = list(ParameterGrid(
    dict(n_epochs=[200], batch_size=[256], with_early_stopping=[True])))

drug_features_extractor_params = list(ParameterGrid(
    dict(arch=['cnn'],
         vocab_size=[len(SMILES_ALPHABET)+2],
         embedding_size=[20],
         cnn_sizes=[
             [128 for _ in range(4)]
         ],
         kernel_sizes=[[5], ],
         dilatation_rate=[1],
         pooling_len=[2],
         b_norm=[False])
))


model_params = list(ParameterGrid(dict(
    model_name=['bmnddi'],
    drug_feature_extractor_params=drug_features_extractor_params,
    fc_layers_dim=[[128]*4],
    mode= ["concat"],
    dropout=[0],
    b_norm=[True],
)))


@click.command()
@click.option('-d', '--dataset', type=str, default='twosides', help='folder to scan')
@click.option('-a', '--algo', type=str, default='bmnddi')
@click.option('-o', '--outfile', type=str, default='configs.json')
def generate_config_file(dataset, algo, outfile):
    #todo: use argument of the function

    expt_config = dict(
        model_params=model_params,
        dataset_params=dataset_params,
        fit_params=fit_params
    )

    with open(outfile, 'w') as f:
        json.dump(expt_config, f, indent=2)

if __name__ == '__main__':
    generate_config_file()
