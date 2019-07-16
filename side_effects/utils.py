import ast
import pickle
from functools import partial

from ivbase.utils.datasets.datacache import DataCache
from sklearn.model_selection import train_test_split

from side_effects.external_utils.utils import *
from side_effects.models.model import *
from side_effects.models.training import DDIModel, compute_metrics
from side_effects.preprocess.dataset import load_train_test_files, load_dataset, to_tensor


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
		# This function return is not used by the train script. But you could do anything with that.
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
    # seed
    dataset = expt_params["dataset"]["name"]
    smiles_transformer = get_transformer(expt_params["dataset"]["smi_transf"])
    rstate = expt_params["dataset"]["seed"]
    mode = expt_params["dataset"]["mode"]

    if rstate not in ('None', None):
        x, y = load_dataset(cach_path, dset_name=dataset)
        # print(y.sum(axis=0))
        #
        # e = np.sum(y, axis=0)
        # c = [int(i / y.shape[0] * 100) for i in e]
        # print(max(c))
        # # Just keep labels that are around [0, 20]
        # labels = []
        # for i, j in enumerate(c):
        #     if j >= 10:
        #         labels.append(i)
        # y = y[:, labels]
        # print("final", y.shape)
        # x, y = x[:100], y[:100, :]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rstate)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=rstate)
        a = compute_classes_weight(y_train)
        print(np.where(a == 1))
        b = non_zeros_distribution(y_train)
        print(b)
        print(a)

    else:
        # load train and test files
        inv_seed = expt_params["dataset"]["inv_seed"]
        targets, x_train, x_test, x_valid, y_train, y_test, y_valid = load_train_test_files(input_path=f"{cach_path}",
                                                                                            dataset_name=dataset,
                                                                                            transformer=smiles_transformer,
                                                                                            seed=inv_seed)
    y_train, y_test, y_valid = list(map(partial(to_tensor, gpu=gpu), [y_train, y_test, y_valid]))
    print(y_train.shape, y_test.shape, y_valid.shape)

    # The loss function
    weights = expt_params["weights"]
    print(weights)
    loss_fn = get_loss(expt_params["loss_function"], y_train=y_train, **weights)
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
                    output_dim=y_train.shape[1], use_gpu=gpu, mode=mode, **expt_params["fc_reg"])
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
    model.train(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, **expt_params["train_params"])
    save(expt_params, "configs.json", output_path)
    save(model.history, "history.json", output_path)
    model.save(os.path.join(output_path, "weights.json"))

    # Test and save
    y_true, y_probs = model.test(x_test, y_test)
    pickle.dump(y_true, open(os.path.join(output_path, "true_labels.pkl"), "wb"))
    pickle.dump(y_probs, open(os.path.join(output_path, "predicted_labels.pkl"), "wb"))
    output = compute_metrics(y_true, y_probs)
    pickle.dump(output, open(os.path.join(output_path, "output.pkl"), "wb"))

# dispatcher -n "ddi_deepddi" -e "deepddi-real-drugbank-inv-seeds" -x 86400 -v 100 -t "ml.c4.8xlarge" -i "s3://datasets-ressources/DDI/INV_DRUGB_REAL" -o  "s3://rogia/EXPT-RES/DEEPDDI/INVIVO/DRUGBANK"  -p ex_configs_1.json -c 9
