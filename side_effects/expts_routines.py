import os
import json
import torch
import pickle
from ivbase.utils.datasets.datacache import DataCache
from side_effects.trainer import Trainer, compute_metrics
from side_effects.data.loader import get_data_partitions


def save(obj, filename, output_path):
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(obj, CNF)


def run_experiment(model_params, dataset_params, fit_params, input_path, output_path="expts"):
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

    dc = DataCache()
    cach_path = dc.get_dir(dir_path="s3://datasets-ressources/DDI/twosides", force=True)
    expt_params = model_params

    print(f"Input folder: {cach_path}")
    print(f"Files in {cach_path}, {os.listdir(cach_path)}")
    print(f"Config params: {expt_params}\n")

    train_data, valid_data, test_data = get_data_partitions(**dataset_params, input_path=cach_path)
    model_params['network_params'].update(dict(output_dim=train_data.nb_labels, use_gpu=torch.cuda.is_available()))
    model = Trainer(**model_params)

    # Train and save
    training = "\n".join([f"{i}:\t{v}" for (i, v) in fit_params.items()])
    print(f"Training details: \n{training}")
    model.train(train_data, valid_data, **fit_params)
    save(expt_params, "configs.json", output_path)
    save(model.history, "history.json", output_path)
    model.save(os.path.join(output_path, "weights.json"))

    # Test and save
    y_true, y_probs = model.test(test_data)
    pickle.dump(y_true, open(os.path.join(output_path, "true_labels.pkl"), "wb"))
    pickle.dump(y_probs, open(os.path.join(output_path, "predicted_labels.pkl"), "wb"))
    output = compute_metrics(y_true, y_probs)
    pickle.dump(output, open(os.path.join(output_path, "output.pkl"), "wb"))
