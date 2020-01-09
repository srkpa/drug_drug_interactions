import hashlib
import json
import os
import pickle
import warnings
from collections import MutableMapping
from itertools import chain

import side_effects.models.mhcaddi.data_download as dd
import side_effects.models.mhcaddi.data_preprocess as dp
import side_effects.models.mhcaddi.split_cv_data as cv
import side_effects.models.mhcaddi.test as test
import side_effects.models.mhcaddi.train as train
from side_effects.data.loader import get_data_partitions, compute_classes_weight, compute_labels_density
from side_effects.models.deep_rf import DeepRF
from side_effects.models.rgcn.link_predict import main
from side_effects.trainer import Trainer

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'

warnings.filterwarnings("ignore")


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
        result_2_filename="{}/{}_{}_{}-res.pkl".format(output_path, data_name, model_name, out_prefix),
        targets_filename="{}/{}_{}_{}_targets.pkl".format(output_path, data_name, model_name, out_prefix),
        preds_filename="{}/{}_{}_{}_preds.pkl".format(output_path, data_name, model_name, out_prefix),
        targets_2_filename="{}/{}_{}_{}-targets.pkl".format(output_path, data_name, model_name, out_prefix),
        preds_2_filename="{}/{}_{}_{}-preds.pkl".format(output_path, data_name, model_name, out_prefix)
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
    # pr = cProfile.Profile()
    # pr.enable()
    all_params = locals()
    del all_params['output_path'], all_params['input_path']
    paths, output_prefix = get_all_output_filenames(output_path, all_params)
    paths["checkpoint_path"] = checkpoint_path

    cach_path = str(os.environ["INVIVO_CACHE_ROOT"]) + "/datasets-ressources/DDI/" + str(
        dataset_params.get('dataset_name'))
    # dc = DataCache()
    # cach_path = dc.sync_dir(dir_path="s3://datasets-ressources/DDI/{}".format(
    #     dataset_params.get('dataset_name')))
    expt_params = model_params
    print(f"Input folder: {cach_path}")
    print(f"Files in {cach_path}, {os.listdir(cach_path)}")
    print(f"Config params: {expt_params}\n")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Restore path if any: {restore_path}")
    save_config(all_params, paths.pop('config_filename'))
    debug = dataset_params.pop("debug", False)
    train_data, valid_data, test_data, unseen_test_data = get_data_partitions(**dataset_params,
                                                                              input_path=cach_path)  # unseen_test_data

    model_name = model_params['network_params'].get('network_name')
    # Variable declaration
    targets, preds, test_perf = {}, {}, {}
    uy_true, uy_probs, u_output = {}, {}, {}
    # Params processing
    if model_name == "mhcaddi":
        dataset_name = dataset_params.get('dataset_name', 'twosides')
        networks_params = model_params.pop("network_params")
        networks_params.pop("network_name")
        if not os.path.exists(cach_path + "/drug_raw_feat.idx.jsonl"):
            dd.main(path=cach_path, dataset_name=dataset_name)
        if not (os.path.exists(cach_path + "/drug.bond_idx.wo_h.self_loop.json") and os.path.exists(
                cach_path + "/drug.feat.wo_h.self_loop.idx.jsonl")):
            dp.main(path=cach_path, dataset_name=dataset_name)

        if debug:
            a = b = c = 20
        else:
            a, b, c = len(train_data.samples), len(valid_data.samples), len(test_data.samples)
        ddis_data = sorted(train_data.samples)[:a] + sorted(valid_data.samples)[:b] + sorted(test_data.samples)[:c]
        train_dataset, valid_dataset, test_dataset = list(
            map(cv.prepare_twosides_dataset,
                [sorted(train_data.samples)[:a], sorted(valid_data.samples)[:b], sorted(test_data.samples)[:c]]))

        n_atom_type, n_bond_type, graph_dict, n_side_effect, side_effect_idx_dict = cv.main(
            path=cach_path,
            ddi_data=ddis_data,
            dataset_name=dataset_name,
            n_fold=3,
            debug=False)

        train.main(n_side_effect=n_side_effect, exp_prefix="", dataset_name=dataset_name,
                   n_atom_type=n_atom_type,
                   train_dataset=train_dataset,
                   test_dataset=test_dataset,
                   valid_dataset=valid_dataset,
                   n_bond_type=n_bond_type, graph_dict=graph_dict,
                   side_effect_idx_dict=side_effect_idx_dict,
                   result_csv_file=paths.get("log_filename"),
                   input_data_path=cach_path, model_dir=output_path, result_dir=output_path,
                   setting_dir=output_path, best_model_pkl=paths.get("checkpoint_filename", None),
                   **fit_params, **networks_params, **model_params)
        # need to recreate the model
        networks_params = {k: v for k, v in networks_params.items() if k in test.main.__code__.co_varnames}
        # Test
        ddis, seidx, score, label = test.main(positive_data=test_dataset['pos'], negative_data=test_dataset['neg'],
                                              dataset_name=dataset_name,
                                              graph_dict=graph_dict, side_effect_idx_dict=side_effect_idx_dict,
                                              n_atom_type=n_atom_type,
                                              n_bond_type=n_bond_type,
                                              n_side_effect=n_side_effect,
                                              best_model_pkl=paths.get("checkpoint_filename", None),
                                              **networks_params)
        #  Reformat final res
        side_effect_idx_dict = {idx: sid for sid, idx in side_effect_idx_dict.items()}
        ddis = list(chain.from_iterable(ddis))
        seidx = [side_effect_idx_dict[idx] for idx in seidx]
        output = list(zip(ddis, seidx, score, label))
        print("TOTAL SCORES. ", len(output), len(output[0]))
        pickle.dump(output, open(paths.get('result_filename'), "wb"))

    elif model_name == 'RGCN':
        output = main(train_data, test_data, valid_data, paths.get("checkpoint_filename", None), model_params,
                      fit_params, debug=False)
        pickle.dump(output, open(paths.get('result_filename'), "wb"))
    elif model_name == "deeprf":
        targets, preds, test_perf = DeepRF(**model_params)(train_data, valid_data, test_data)
    else:
        # Set up of loss function params
        loss_params = model_params["loss_params"]
        model_params["loss_params"]["weight"] = compute_classes_weight(train_data.get_targets()) if \
            loss_params["use_fixed_binary_cost"] else None
        model_params["loss_params"]["density"] = compute_labels_density(train_data.get_targets()) if \
            loss_params["use_fixed_label_cost"] else None
        del (model_params["loss_params"]["use_fixed_binary_cost"])
        del (model_params["loss_params"]["use_fixed_label_cost"])

        # Configure the auxiliary network who process additional features
        if model_params['network_params'].get('auxnet_params', None):
            model_params['network_params']["auxnet_params"]["input_dim"] = train_data.get_aux_input_dim()

        if model_name == "adnn":
            train_data, valid_data, model_params = fit_encoders(model_params=model_params, train_data=train_data,
                                                                valid_data=valid_data, fit_params=fit_params)
        # Train and save
        model_params['network_params'].update(dict(output_dim=train_data.nb_labels))
        model = Trainer(**model_params, snapshot_dir=restore_path)
        print(model.model)
        model.cuda()
        training = "\n".join([f"{i}:\t{v}" for (i, v) in fit_params.items()])
        print(f"Training details: \n{training}")
        model.train(train_data, valid_data, **fit_params, **paths)
        # Test and save
        targets, preds, test_perf = model.test(test_data)
        uy_true, uy_probs, u_output = model.test(unseen_test_data)

        # Save model  test results
        pickle.dump(targets, open(paths.get('targets_filename'), "wb"))
        pickle.dump(preds, open(paths.get('preds_filename'), "wb"))
        pickle.dump(test_perf, open(paths.get('result_filename'), "wb"))
        pickle.dump(uy_true, open(paths.get('targets_2_filename'), "wb"))
        pickle.dump(uy_probs, open(paths.get('preds_2_filename'), "wb"))
        pickle.dump(u_output, open(paths.get('result_2_filename'), "wb"))
    # pr.disable()
    # pr.print_stats()
    # pr.dump_stats(os.path.join(output_path, "profiling_result.txt"))


def fit_encoders(model_params, fit_params, train_data, valid_data):
    assert len(train_data) == 4
    assert len(valid_data) == 4
    mm_hidden_layer_dims = model_params["network_params"].pop("mn_hidden_layer_dims")
    network_name = model_params["network_params"].get("network_name")
    model_params["network_params"]["input_dim"] = train_data[0].get_samples()[0][0].shape[-1]
    encoders = []
    for i in range(3):
        encoder = Trainer(**model_params)
        encoder.cuda()
        encoder.train(train_data[i], valid_data[i], **fit_params)
        model_params["network_params"]["network_name"] = network_name
        encoders.append(encoder)
    encoders = dict(zip(["drug_encoder", "gen_encoder", "func_encoder"], encoders))
    del model_params["network_params"]
    model_params.update({"network_params": dict(network_name='dnn', gen_encoder=encoders["gen_encoder"],
                                                drug_encoder=encoders["drug_encoder"],
                                                func_encoder=encoders["func_encoder"],
                                                hidden_layer_dims=[2000] * 6,
                                                activation='relu',
                                                dropout=0.3,
                                                b_norm=True,
                                                ),
                         "metrics_names": [
                             "macro_roc",
                             "macro_auprc",
                             "micro_roc",
                             "micro_auprc"
                         ]
                         })
    model_params.update(dict(optimizer='adam', lr=1e-4, loss='bce'))
    train_data, valid_data = train_data[-1], valid_data[-1]
    return train_data, valid_data, model_params
