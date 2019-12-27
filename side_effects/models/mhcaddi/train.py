"""
	File used to train the networks.
"""
import csv
import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

from .ddi_dataset import ddi_collate_paired_batch, \
    PolypharmacyDataset, ddi_collate_batch
from .qm9_dataset import QM9Dataset, qm9_collate_batch
from .utils.ddi_utils import ddi_train_epoch, ddi_valid_epoch
from .utils.file_utils import setup_running_directories
from .utils.functional_utils import combine
from .utils.qm9_utils import qm9_train_epoch, qm9_valid_epoch


def post_parse_args(opt):
    # Set the random seed manually for reproducibility.
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    if not hasattr(opt, 'exp_prefix'):
        opt.exp_prefix = opt.memo + '-cv_{}_{}'.format(opt.fold_i, opt.n_fold)
    if opt.debug:
        opt.exp_prefix = 'dbg-{}'.format(opt.exp_prefix)
    if not hasattr(opt, 'global_step'):
        opt.global_step = 0

    opt.best_model_pkl = os.path.join(opt.model_dir, opt.exp_prefix + '.pth')
    opt.result_csv_file = os.path.join(opt.result_dir, opt.exp_prefix + '.csv')

    print(opt.best_model_pkl)
    print(opt.result_csv_file)
    print(opt.model_dir)
    return opt


def prepare_qm9_dataloaders(opt):
    train_loader = torch.utils.data.DataLoader(
        QM9Dataset(
            graph_dict=opt.graph_dict,
            pairs_dataset=opt.train_dataset),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=qm9_collate_batch,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        QM9Dataset(
            graph_dict=opt.graph_dict,
            pairs_dataset=opt.valid_dataset),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=qm9_collate_batch)
    return train_loader, valid_loader


def prepare_ddi_dataloaders(graph_dict, side_effect_idx_dict, train_dataset, train_neg_pos_ratio, batch_size,
                            valid_dataset):
    train_loader = torch.utils.data.DataLoader(
        PolypharmacyDataset(
            drug_structure_dict=graph_dict,
            se_idx_dict=side_effect_idx_dict,
            se_pos_dps=train_dataset['pos'],
            # TODO: inspect why I'm not just fetching opt.train_dataset['neg']
            negative_sampling=True,
            negative_sample_ratio=train_neg_pos_ratio,
            paired_input=True,
            n_max_batch_se=10),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=ddi_collate_paired_batch,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        PolypharmacyDataset(
            drug_structure_dict=graph_dict,
            se_idx_dict=side_effect_idx_dict,
            se_pos_dps=valid_dataset['pos'],
            se_neg_dps=valid_dataset['neg'],
            n_max_batch_se=1),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=lambda x: ddi_collate_batch(x, return_label=True))
    return train_loader, valid_loader


def train_epoch(model, data_train, optimizer, averaged_model, device, dataset_name, transH, learning_rate, global_step):
    if dataset_name == "qm9":
        opt = {}
        return qm9_train_epoch(model, data_train, optimizer, averaged_model, device, opt)
    else:
        return ddi_train_epoch(model, data_train, optimizer, averaged_model, device, transH, learning_rate, global_step)


def valid_epoch(model, data_valid, device, dataset_name):
    if dataset_name == "qm9":
        opt = {}
        return qm9_valid_epoch(model, data_valid, device, opt)
    else:
        return ddi_valid_epoch(model, data_valid, device)


def train(model, datasets, device, dataset_name, l2_lambda, learning_rate, result_csv_file, n_epochs, global_step,
          best_model_pkl, patience, transH):
    data_train, data_valid = datasets
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    with open(result_csv_file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        if dataset_name == "twosides":
            csv_writer.writerow(['train_loss', 'auroc_valid'])
        if dataset_name == "qm9":
            csv_writer.writerow(['train_loss', 'auroc_valid', 'individual_maes'])

    ddi_best_valid_perf = 0
    qm9_best_valid_perf = 10
    waited_epoch = 0
    averaged_model = model.state_dict()
    for epoch_i in range(n_epochs):
        print()
        print('Epoch ', epoch_i)

        # ============= Training Phase =============
        train_loss, elapse, averaged_model = \
            train_epoch(model, data_train, optimizer, averaged_model, device, dataset_name, transH, learning_rate,
                        global_step)
        logging.info('  Loss:       %5f, used time: %f min', train_loss, elapse)

        # ============= Validation Phase =============

        # Load the averaged model weight for validation
        updated_model = model.state_dict()  # validation start
        model.load_state_dict(averaged_model)

        valid_perf, elapse = valid_epoch(model, data_valid, device, dataset_name)
        # AUROC is in fact MAE for QM9
        valid_auroc = valid_perf['auroc']
        if dataset_name == "qm9":
            individual_maes = valid_perf['individual_maes']
        logging.info('  Validation: %5f, used time: %f min', valid_auroc, elapse)
        # print_performance_table({k: v for k, v in valid_perf.items() if k != 'threshold'})

        # Load back the trained weight
        model.load_state_dict(updated_model)  # validation end

        # early stopping
        if (dataset_name == "twosides" and valid_auroc > ddi_best_valid_perf) \
                or (dataset_name == "qm9" and valid_auroc < qm9_best_valid_perf):
            logging.info('  --> Better validation result!')
            waited_epoch = 0
            torch.save(
                {'global_step': global_step,
                 'model': averaged_model,
                 'threshold': valid_perf['threshold']},
                best_model_pkl)
        else:
            if waited_epoch < patience:
                waited_epoch += 1
                logging.info('  --> Observing ... (%d/%d)', waited_epoch, patience)
            else:
                logging.info('  --> Saturated. Break the training process.')
                break

        # ============= Bookkeeping Phase =============
        # Keep the validation record
        if dataset_name == "twosides":
            ddi_best_valid_perf = max(valid_auroc, ddi_best_valid_perf)
        if dataset_name == "qm9":
            qm9_best_valid_perf = min(valid_auroc, qm9_best_valid_perf)

        # Keep all metrics in file
        with open(result_csv_file, 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            if dataset_name == "twosides":
                csv_writer.writerow([train_loss, valid_auroc])
            if dataset_name == "qm9":
                csv_writer.writerow([train_loss, valid_auroc, individual_maes])


def main(input_data_path, n_side_effect, side_effect_idx_dict, n_atom_type, n_bond_type, graph_dict, exp_prefix,
         dataset_name, result_csv_file,
         model_dir, result_dir, setting_dir, fold_i=3, batch_size=128, d_hid=32, d_atom_feat=3, n_prop_step=3,
         score_fn='trans',
         n_attention_head=1, n_epochs=10000, d_readout=32, learning_rate=1e-3, l2_lambda=0, dropout=0.1,
         patience=8, train_neg_pos_ratio=1, transR=False, transH=False, trained_setting_pkl=None, best_model_pkl=None,
         global_step=0):
    params_dict = locals()
    is_resume_training = trained_setting_pkl is not None
    if is_resume_training:
        logging.info('Resume training from', trained_setting_pkl)
        opt = np.load(open(trained_setting_pkl, 'rb'), allow_pickle=True).item()
    else:
        setup_running_directories(setting_dir, model_dir, result_dir)
        # save the dataset split in setting dictionary
        # pprint.pprint(vars(opt))
        # save the setting
        logging.info('Related data will be saved with prefix: %s', exp_prefix)

    # assert os.path.exists(opt.input_data_path + "folds/")

    # code which is common for ddi and qm9. take care(P0) if adding other datasets
    # data_opt = np.load(open(opt.input_data_path + "input_data.npy", 'rb'), allow_pickle=True).item()

    print(len(graph_dict))
    if dataset_name == "twosides":
        # 'pos'/'neg' will point to a dictionary where
        # each se points to a list of drug-drug pairs.
        train_dataset = {'pos': {}, 'neg': {}}
        test_dataset = pickle.load(open(input_data_path + "/" + str(fold_i) + "fold.npy", "rb"))
        valid_fold = fold_i - 1
        valid_dataset = pickle.load(open(input_data_path + "/" + str(valid_fold) + "fold.npy", "rb"))

        for i in range(1, valid_fold + 1):
            if i != fold_i:
                dataset = pickle.load(open(input_data_path + "/" + str(i) + "fold.npy", "rb"))
                train_dataset['pos'] = combine(train_dataset['pos'], dataset['pos'])
                train_dataset['neg'] = combine(train_dataset['neg'], dataset['neg'])

        # assert data_opt.n_side_effect == len(opt.side_effects)
        dataloaders = prepare_ddi_dataloaders(graph_dict=graph_dict, side_effect_idx_dict=side_effect_idx_dict,
                                              train_dataset=train_dataset, train_neg_pos_ratio=train_neg_pos_ratio,
                                              batch_size=batch_size, valid_dataset=valid_dataset)

    # save_experiment_settings(opt)

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("using cuda")
    else:
        print("on cpu")

    if transR:
        from .model_r import DrugDrugInteractionNetworkR as DrugDrugInteractionNetwork
    elif transH:
        from .model_h import DrugDrugInteractionNetworkH as DrugDrugInteractionNetwork
    else:
        from .model import DrugDrugInteractionNetwork

    model = DrugDrugInteractionNetwork(
        n_side_effect=n_side_effect,
        n_atom_type=n_atom_type,
        n_bond_type=n_bond_type,
        d_node=d_hid,
        d_edge=d_hid,
        d_atom_feat=d_atom_feat,
        d_hid=d_hid,
        d_readout=d_readout,
        n_head=n_attention_head,
        n_prop_step=n_prop_step,
        dropout=dropout,
        score_fn=score_fn).to(device)

    if is_resume_training:
        trained_state = torch.load(best_model_pkl)
        global_step = trained_state['global_step']
        logging.info(
            'Load trained model @ step %d from file: %s',
            global_step, best_model_pkl)
        model.load_state_dict(trained_state['model'])

    train(model, dataloaders, device, dataset_name, l2_lambda, learning_rate, result_csv_file, n_epochs, global_step,
          best_model_pkl, patience, transH)
    params_dict.update({"graph_dict": graph_dict})
    return test_dataset
