import os
import json
import ast
import torch.optim as optim
from sklearn.utils import compute_class_weight
from side_effects.preprocess.dataset import load_train_test_files
from side_effects.external_utils.loss import Weighted_binary_cross_entropy1, Weighted_cross_entropy, mloss
from ivbase.nn.extractors.cnn import Cnn1dFeatExtractor
from side_effects.models.model import DDIModel, DDINetwork, DeepDDI, PCNN

from side_effects.models.test import Cnn1d

from side_effects.external_utils.init import *
from ivbase.utils.gradcheck import *
from ivbase.utils.gradinspect import *
from ivbase.utils.datasets.datacache import DataCache
from side_effects.preprocess.transforms import fingerprints_transformer, sequence_transformer


def compute_classes_weight(y):
    return torch.tensor([compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=target)
       if len(np.unique(target)) > 1 else np.array([1.0, 1.0]) for target in y.T], dtype=torch.float32).t()


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

    print("model_params", model_params)

    # Load the appropriate folder from s3
    if input_path.startswith("/opt/ml"):
        cach_path = input_path
        train_params = {arg: ast.literal_eval(val) if arg not in ['dataset_name', "method"] else val for arg, val in
                        model_params.items()}
    else:
        dc = DataCache()
        cach_path = dc.get_dir(dir_path=input_path)
        train_params = model_params

    print("train_params", train_params)
    print(cach_path, os.listdir(cach_path))

    method = train_params["method"]
    dataset_name = train_params["dataset_name"]
    batch_size = train_params["fit_params"]["batch_size"]
    lr = train_params["fit_params"]["lr"]
    n_epochs = train_params["fit_params"]["n_epochs"]
    early_stopping = train_params["fit_params"]["with_early_stopping"]
    thresholds = [i/100 for i in range(101)]

    # load train and test files
    _, x_train, x_test, x_val, y_train, y_test, y_val = load_train_test_files(input_path=f"{cach_path}",
                                                                              dataset_name=dataset_name,
                                                                              method=method,
                                                                              transformer=fingerprints_transformer)
    x_train = torch.stack([torch.cat(pair) for pair in x_train])
    print(x_train.shape)
    x_test = torch.stack([torch.cat(pair) for pair in x_test])
    print(x_test.shape)
    x_val = torch.stack([torch.cat(pair) for pair in x_val])
    print(x_val.shape)

    # w = compute_classes_weight(y=y_train)
    # print("weight", w.shape)
    # print("weight", w)
    #
    # # Loss function
    # if torch.cuda.is_available():
    #     w = w.cuda()
    #
    # loss_function = Weighted_binary_cross_entropy1(weights_per_targets=w)
    # print("loss_function", loss_function)
    loss_function = nn.BCEWithLogitsLoss()
    print("loss_function", loss_function)
    if method == "deepddi":
        net = DeepDDI(input_dim=100, output_dim=y_test.shape[1], hidden_sizes=[2048] * 9)
        net.apply(init_uniform)
    else:

        # Set hps and params
        use_targets = train_params["classifier"]["use_targets"]
        hidden_sizes = train_params["classifier"]["hidden_sizes"]
        extractor_params = train_params['feature_extractor_params']

        params = dict(
            drugs_feature_extractor=Cnn1dFeatExtractor(**extractor_params),
            hidden_sizes=hidden_sizes,
            use_tar=use_targets
        )

        # Build the network
        net = DDINetwork(**params, output_dim=y_test.shape[1])
        # net = PCNN(150, 20, [32]*3,  [3, 5, 7])
    # Next step: Choose the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Build the pytoune model
    model = DDIModel(net=net, lr=lr, loss=loss_function, optimizer=optimizer)
    print(model.model)

    # Move to gpu
    if torch.cuda.is_available():
        model = model.cuda()
        x_train = x_train.cuda()
        x_test = x_test.cuda()
        x_val = x_val.cuda()
        y_train = torch.from_numpy(y_train).cuda()
        y_val = torch.from_numpy(y_val).cuda()
        y_test = torch.from_numpy(y_test).cuda()

    # Train and save
    model.train(x_train, y_train, x_val, y_val, n_epochs=n_epochs, batch_size=batch_size,
                with_early_stopping=early_stopping)

    with open(os.path.join(output_path, "config.json"), 'w') as CNF:
        json.dump(train_params, CNF)

    with open(os.path.join(output_path, "history.json"), 'w') as HIST:
        json.dump(model.history, HIST)

    model.save(os.path.join(output_path, "weights.json"))

    # Test and save metrics
    y_true, y_pred = model.test(x_test, y_test)

    for threshold in thresholds:
        expts_out = model.cmetrics(y_true, y_pred, threshold=threshold, dataset_name=dataset_name)
        with open(os.path.join(output_path, f"{threshold}_out.json"), 'w') as OUT:
            json.dump(expts_out, OUT)

#dispatcher -n "side_effects_label_weight" -e "expt-testML" -x 86400 -v 100 -t "ml.p2.xlarge" -i "s3://datasets-ressources/DDI/drugbank" -o  "s3://invivoai-sagemaker-artifacts/ddi"  -p configs_3.json
#side_effect_finger_no_bfc