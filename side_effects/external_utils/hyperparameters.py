import json
from sklearn.model_selection import ParameterGrid


def hps_setter(params_format, params_to_change, save):
    model_params = params_format
    for key, val in model_params.items():
        if key in params_to_change:
            model_params[key] = params_to_change[key]
        else:
            if isinstance(val, dict):
                for nested_key, _ in val.items():
                    if nested_key in params_to_change:
                        model_params[key][nested_key] = params_to_change[nested_key]

    with open(save, "w") as OUT:
        json.dump(model_params, OUT)

    return model_params


def generate_config_file(template, grid, output_path):
    params = list(ParameterGrid(grid))
    print(len(params))
    for i, elem in enumerate(params):
        expt_name = "expt-" + '%'.join(
            list(elem.keys()) + [str(val) if not isinstance(val, list) else str(len(val)) for val in elem.values()])
        print(expt_name)
        elem["expt_name"] = expt_name
        hps_setter(params_format=template, params_to_change=elem, save=f"{output_path}/{i}_config.json")


def config_file(params, config_path="../../expts/configs.json"):
    with open(config_path, "w") as cp:
        json.dump(params, cp)


if __name__ == '__main__':
    # noublie pas le transpose dans loss weightes
    import json
    import os
    filename = "/home/rogia/Documents/git/side_effects/expts/ex_configs_2.json"
    configs = json.load(open(filename, "r"))
    kernel_sizes = [3]
    extractor = configs["extractor_params"][-1]
    losses = ["weighted"]
    initializations = ["constant",
                       "kaiming_uniform",
                       "normal",
                       "uniform",
                       "xavier"]
    print(kernel_sizes, extractor)
    extractors = []
    for ks in kernel_sizes:
        a = {k:v if k != "kernel_size" else ks for k, v in extractor.items()}
        extractors.append(a)
    print(extractors)
    configs["init_fn"].extend(initializations)
    configs["loss_function"] = losses
    configs["extractor_params"] = extractors

    output_path = "/home/rogia/Documents/git/side_effects/expts/"
    filename = "configs.json"
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(configs, CNF)

    exit()

    template = dict(dataset_name=["drugbank"], method=["DRUUD"],
                    fit_params=[{'n_epochs': 100, 'lr': 1e-04, 'batch_size': 256, 'with_early_stopping': True}],
                    feature_extractor_params=[{"vocab_size": 151, "embedding_size": 151, "cnn_sizes": [84],
                                               "kernel_size": 6, "dropout": 0, "pooling": "max", "b_norm": True,
                                               "pooling_len": 2, "normalize_features": True},
                                              {'vocab_size': 151, 'embedding_size': 151,
                                               'cnn_sizes': [32, 64, 128, 256, 512],
                                               'kernel_size': [11, 9, 7, 5, 3], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151, 'cnn_sizes': [32, 64],
                                               'kernel_size': [5, 3], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151, 'cnn_sizes': [32, 64, 12],
                                               'kernel_size': [7, 5, 3], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151,
                                               'cnn_sizes': [32, 64, 128, 256],
                                               'kernel_size': [9, 7, 5, 3], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151, 'cnn_sizes': [32],
                                               'kernel_size': [3], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151, 'cnn_sizes': [32, 64],
                                               'kernel_size': [3, 5], 'dropout': 0.0, 'pooling': 'max',
                                               'use_bnb_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151, 'cnn_sizes': [32, 64, 128],
                                               'kernel_size': [3, 5, 7], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151,
                                               'cnn_sizes': [32, 64, 128, 256],
                                               'kernel_size': [3, 5, 7, 9], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True},
                                              {'vocab_size': 151, 'embedding_size': 151,
                                               'cnn_sizes': [32, 64, 128, 256, 512],
                                               'kernel_size': [3, 5, 7, 9, 11], 'dropout': 0.0, 'pooling': 'max',
                                               'b_norm': True,
                                               'pooling_len': 2, 'normalize_features': True}
                                              ],
                    classifier=[{'hidden_sizes': [512], 'use_targets': False}],
                    test=[{"thresholds": [0.47, 0.5]}])

    template_2 = dict(dataset_name=["drugbank"], method=["DRUUD"],
                      fit_params=[{'n_epochs': 100, 'lr': 1e-04, 'batch_size': 256, 'with_early_stopping': True}],
                      feature_extractor_params=[{"vocab_size": 150, "embedding_size": 20, "cnn_sizes": [128],
                                                 "kernel_size": 3, "dropout": 0.0, "pooling": "max", "b_norm": True,
                                                 "pooling_len": 2, "normalize_features": True}],
                      classifier=[{'hidden_sizes': [512], 'use_targets': False}],
                      test=[{"thresholds": [0.47, 0.5]}])

    config_file(params=template_2, config_path="../../expts/ex_configs_2.json")
    config_file(params=template_2, config_path="../../expts/configs_2.json")
