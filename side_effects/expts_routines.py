import os
import json
import torch
import pickle
import hashlib
from collections import MutableMapping, OrderedDict
from side_effects.trainer import Trainer
from side_effects.models.deep_rf import DeepRF
from side_effects.data.loader import get_data_partitions, DataCache, compute_classes_weight, compute_labels_density

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_outfname_prefix(all_params):
    params_flatten = flatten_dict(all_params)
    base = str(params_flatten)
    uid = hashlib.md5(str(base).encode()).hexdigest()[:8]
    return uid


def get_all_output_filenames(output_path, all_params):
    out_prefix = get_outfname_prefix(all_params)
    model_name = all_params.get('model_params').get('network_params').get('network_name')
    data_name = all_params.get('dataset_params').get('dataset_name')
    return dict(
        config_filename="{}/{}_{}_{}_params.json".format(output_path, data_name, model_name, out_prefix),
        checkpoint_filename="{}/{}_{}_{}.checkpoint.pth".format(output_path, data_name, model_name, out_prefix),
        log_filename="{}/{}_{}_{}_log.log".format(output_path, data_name, model_name, out_prefix),
        tensorboard_dir="{}/{}_{}_{}".format(output_path, data_name, model_name, out_prefix),
        result_filename="{}/{}_{}_{}_res.pkl".format(output_path, data_name, model_name, out_prefix),
        targets_filename="{}/{}_{}_{}_targets.pkl".format(output_path, data_name, model_name, out_prefix),
        preds_filename="{}/{}_{}_{}_preds.pkl".format(output_path, data_name, model_name, out_prefix),
    ), out_prefix


def save_config(all_params, outfile):
    with open(outfile, 'w') as fd:
        temp = flatten_dict(all_params)
        temp = {k: temp[k] for k in temp
                if isinstance(temp[k], (str, int, float, bool, list, dict, tuple, type(None)))}
        json.dump(temp, fd, indent=4, sort_keys=True)


def save(obj, filename, output_path):
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(obj, CNF)


def run_experiment(model_params, dataset_params, input_path, output_path, restore_path,
                   checkpoint_path, fit_params=None):
    """
    Placeholder function that should be implemented to run the models


    Parameters
    ----------
    model_params : dict
        The model parameter, loaded from a json file. This corresponds to the parameters
        that the train script will send you.

    dataset_params : dict
        The dataset parameter, loaded from a json file. This corresponds to the parameters
        that the train script will send you.

    fit_params : dict
        The fitting parameter, loaded from a json file. This corresponds to the parameters
        that the train script will send you.

    output_path : str
        Path to the Location where all training results will be saved.
        This will be provided to you by the train script

    input_path : str, optional
        Path to the Location where input data are. This will be provided to you by the train script.
        You might discard this, if your data can be found in a path inside your package that you could load yourself.

    **kwargs : optional
        Additional named parameters

    Returns
    -------
        # This function return is not used by the train script. But you could do anything with that.
    """
    all_params = locals()
    del all_params['output_path'], all_params['input_path']
    paths, output_prefix = get_all_output_filenames(output_path, all_params)
    paths["checkpoint_path"] = checkpoint_path

    # cach_path = f"{input_path}/{dataset_params.get('dataset_name')}"
    dc = DataCache()
    cach_path = dc.sync_dir(dir_path="s3://datasets-ressources/DDI/{}".format(
        dataset_params.get('dataset_name')))
    expt_params = model_params

    print(f"Input folder: {cach_path}")
    print(f"Files in {cach_path}, {os.listdir(cach_path)}")
    print(f"Config params: {expt_params}\n")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Restore path if any: {restore_path}")
    save_config(all_params, paths.pop('config_filename'))
    train_data, valid_data, test_data = get_data_partitions(**dataset_params, input_path=cach_path)
    algorithm = model_params['network_params'].get('network_name')
    y_true, y_probs, output = {}, {}, {}
    if algorithm == "deeprf":
        y_true, y_probs, output = DeepRF(**model_params)(train_data, valid_data, test_data)
    else:
        model_params['network_params'].update(dict(output_dim=train_data.nb_labels))
        # Set up of loss function params
        loss_params = model_params["loss_params"]
        model_params["loss_params"]["weight"] = compute_classes_weight(train_data.get_targets()) if \
            loss_params["use_fixed_binary_cost"] else None
        model_params["loss_params"]["density"] = compute_labels_density(train_data.get_targets()) if \
            loss_params["use_fixed_label_cost"] else None
        del (model_params["loss_params"]["use_fixed_binary_cost"])
        del (model_params["loss_params"]["use_fixed_label_cost"])

        # Configure the auxiliary network who process side features
        if model_params['network_params'].get('auxnet_params', None):
            model_params['network_params']["auxnet_params"]["input_dim"] = train_data.get_aux_input_dim()

        # Train and save
        model = Trainer(**model_params, snapshot_dir=restore_path)
        training = "\n".join([f"{i}:\t{v}" for (i, v) in fit_params.items()])
        print(f"Training details: \n{training}")
        model.train(train_data, valid_data, **fit_params, **paths)
        # Test and save
        y_true, y_probs, output = model.test(test_data)

    pickle.dump(y_true, open(paths.get('targets_filename'), "wb"))
    pickle.dump(y_probs, open(paths.get('preds_filename'), "wb"))
    pickle.dump(output, open(paths.get('result_filename'), "wb"))

