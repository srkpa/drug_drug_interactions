from side_effects.utils import *
from side_effects.preprocess.dataset import *
import os
import json
from sklearn.metrics import classification_report, accuracy_score
from side_effects.preprocess.transforms import fingerprints_transformer, sequence_transformer
from side_effects.models.test import Cnn1d


def load_and_test(input_path, output_path, thresholds):
    dir = output_path
    train_params, dataset_name, method, lr = {}, "", "", None
    for file in os.listdir(dir):
        if file.endswith("config.json"):
            with open(os.path.join(dir, file), "r") as tc:
                train_params = json.load(tc)
                method = train_params["method"]
                dataset_name = train_params["dataset_name"]
                lr = train_params["fit_params"]["lr"]

    dc = DataCache()
    cach_path = dc.get_dir(dir_path=input_path)
    print(cach_path)
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

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    if method == "deepddi":
        net = DeepDDI(input_dim=100, output_dim=y_test.shape[1], hidden_sizes=[2048] * 9)
        net.apply(init_uniform)
    else:

        # Set hps and params
        use_targets = train_params["classifier"]["use_targets"]
        hidden_sizes = train_params["classifier"]["hidden_sizes"]
        extractor_params = train_params['feature_extractor_params']

        params = dict(
            drugs_feature_extractor=Cnn1d(**extractor_params),#Cnn1dFeatExtractor(**extractor_params),
            hidden_sizes=hidden_sizes,
            use_tar=use_targets
        )
        # Compute weight and move to gpu if available
        if dataset_name == "twosides":
            loss = train_params["fit_params"]["loss"]
            w = compute_classes_weight(y=y_train)
            if torch.cuda.is_available():
                w = w.cuda()
            # Choose the loss function
            if loss == "weighted":
                loss_function = WeightedBinaryCrossEntropy1(weights_per_batch_element=w)

        # Build the network
        net = DRUUD(**params, output_dim=y_test.shape[1])

    # Next step: Choose the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Build the pytoune model
    model = DDIModel(net=net, lr=lr, loss=loss_function, optimizer=optimizer)
    model.load(os.path.join(dir, "weights.json"))
    model.model.eval()

    print(model.model)

    print("Loading finished.........")
    y_true, y_pred = model.predict_proba(x_test, y_test, batch_size=32)
    print("Testing finished")

    y_pred = np.where(y_pred >= 0.47, 1, 0)
    clr = classification_report(y_true, y_pred)
    print(clr)
    with open(os.path.join(output_path, "report.json"), "w") as rp:
        json.dump(clr, rp)

    for threshold in thresholds:
        expts_out = model.cmetrics(y_true, y_pred, threshold=threshold, dataset_name=dataset_name)
        print(threshold, expts_out)
        with open(os.path.join(output_path, f"{threshold}_out.json"), 'w') as OUT:
            json.dump(expts_out, OUT)


if __name__ == '__main__':
    input_path = "s3://datasets-ressources/DDI/drugbank"
    output_path_1 = "/home/rogia/Documents/git/invivoprojects/side_effects/data/expt-noemb-FGP-2019-05-11-19-04-41-891/output"
    thresholds = [i / 100 for i in range(0, 47, 1)] + [i / 100 for i in range(48, 100, 1)]
    load_and_test(input_path, output_path_1, thresholds)
