import ast
import pickle
import torch
from functools import partial
from ivbase.utils.datasets.datacache import DataCache

from side_effects.external_utils.utils import *
from side_effects.models.model import *
from side_effects.models.training import DDIModel, compute_metrics
from side_effects.preprocess.dataset import load_train_test_files, make_tensor, to_tensor


def run_experiment(model_params, input_path, output_path="expts"):
    """
    Placeholder function that should be implemented to run the models


    Parameters
    ----------
    model_params : Union([dict, list])
        The model parameter, loaded from a json file. This corresponds to the parameters
        that the train script will send you. You can be clever in what to be expected in this file
        and load additional parameters from your package given the input value.

	output_path: str
		Path to the Location where all training results will be saved.
		This will be provided to you by the train script

	input_path: str, optional
		Path to the Location where input data are. This will be provided to you by the train script.
		You might discard this, if your data can be found in a path inside your package that you could load yourself.

	**kwargs: optional
		Additional named parameters

    Returns
    -------
		This function return is not used by the train script. But you could do anything with that.
    """

    items = ["init_fn", "extractor", "arch", "loss_function", "optimizer"]
    # Load the appropriate folder from s3 # i ll be back
    if input_path.startswith("/opt/ml"):
        cach_path = input_path
        expt_params = {arg: ast.literal_eval(val) if arg not in items else val for arg, val in
                       model_params.items()}
    else:
        dc = DataCache()
        cach_path = dc.get_dir(dir_path=input_path)
        expt_params = model_params

    print(f"Input folder: {cach_path}")
    print(f"Files in {cach_path}, {os.listdir(cach_path)}")
    print(f"Config params: {expt_params}\n")

    gpu = False
    if torch.cuda.is_available():
        gpu = True

    # Dataset
    dataset = expt_params["dataset"]["name"]
    smi_transformer = get_transformer(expt_params["dataset"]["smi_transf"])

    # load train and test files
    targets, x_train, x_test, x_val, y_train, y_test, y_val = load_train_test_files(input_path=f"{cach_path}",
                                                                                    dataset_name=dataset,
                                                                                    transformer=smi_transformer)

    x_train, x_test, x_val = list(map(partial(make_tensor, gpu=gpu), [x_train, x_test, x_val]))
    y_train, y_test, y_val = list(map(partial(to_tensor, gpu=gpu), [y_train, y_test, y_val]))

    print(x_train.shape, y_train.shape)
    # The loss function
    loss_fn = get_loss(expt_params["loss_function"], y_train=y_train)
    print(f"Loss Function: {loss_fn}")

    # Initialization
    init_fn = expt_params["init_fn"]
    print(f"Initialization: {init_fn}")

    #  which network
    method = expt_params["arch"]

    # The network + Initializations
    # a) Build the drug feature extractor network
    dg_net = get_network(expt_params["extractor"], expt_params["extractor_params"])
    # b) Build the model
    network = DRUUD(drug_feature_extractor=dg_net, fc_layers_dim=expt_params["fc_layers_dim"],
                    output_dim=y_train.shape[1])
    if init_fn not in ('None', None):
        network.apply(get_init_fn(init_fn))
    print(f"Architecture:\n\tname: {method}\n\ttnetwork:{network}")

    #  The optimizer
    op = expt_params["optimizer"]
    op_params = expt_params["optimizer_params"]
    optimizer = get_optimizer(op, network, op_params)
    print(f"Optimizer:\n\tname: {op}\n\tparams: {op_params}")

    # The pytoune model
    model = DDIModel(network, optimizer, loss_fn)
    if torch.cuda.is_available():
        model = model.cuda()
    # Train and save
    trainin = "\n".join([f"{i}:\t{v}" for (i, v) in expt_params["train_params"].items()])
    print(f"Training details: \n{trainin}")
    model.train(x_train=x_train, y_train=y_train, x_valid=x_val, y_valid=y_val, **expt_params["train_params"])
    save(expt_params, "configs.json", output_path)
    save(model.history, "history.json", output_path)
    model.save(os.path.join(output_path, "weights.json"))

    # Test and save
    y_true, y_probs = model.test(x_test, y_test)
    output = compute_metrics(y_true, y_probs)
    pickle.dump(output, open(os.path.join(output_path, "output.pkl"), "wb"))
