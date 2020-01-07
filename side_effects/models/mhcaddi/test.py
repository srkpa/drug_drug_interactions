import logging

import torch
import torch.utils.data

from .ddi_dataset import PolypharmacyDataset, ddi_collate_batch
from .qm9_dataset import QM9Dataset, qm9_collate_batch
# from drug_data_util import copy_dataset_from_pkl
from .train import valid_epoch as run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

"""
def pair_qm9_test(test_graph_dict, train_graph_dict, test_labels_dict, train_labels_dict):
	train_k_list = list(train_graph_dict.keys())
	test_kv_list = [(k,v) for k,v in test_graph_dict.items()]
	random.shuffle(test_kv_list)
	train_key = random.choice(train_k_list)

	dataset = []
	for i, kv_pair in enumerate(test_kv_list):
		# i-th key in kv_list1
		test_key = kv_pair[0]

		test_lbl = test_labels_dict[test_key]
		train_lbl = train_labels_dict[train_key]
		dataset.append((test_key,train_key,test_lbl,train_lbl))
	return dataset
"""


def prepare_qm9_testset_dataloader(opt):
    test_loader = torch.utils.data.DataLoader(
        QM9Dataset(
            graph_dict=opt.graph_dict,
            pairs_dataset=opt.test_dataset),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=qm9_collate_batch)
    return test_loader


def prepare_ddi_testset_dataloader(positive_set, negative_set, batch_size, drug_dict, side_effect_idx_dict):
    test_loader = torch.utils.data.DataLoader(
        PolypharmacyDataset(
            drug_structure_dict=drug_dict,
            se_idx_dict=side_effect_idx_dict,
            se_pos_dps=positive_set,
            se_neg_dps=negative_set,
            n_max_batch_se=1),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=lambda x: ddi_collate_batch(x, return_label=True))
    return test_loader


def load_trained_model(transR, transH, device, n_side_effect, d_hid, d_readout, n_attention_head, n_prop_step,
                       best_model_pkl, n_atom_type, n_bond_type, d_atom_feat):
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
        n_prop_step=n_prop_step).to(device)
    trained_state = torch.load(best_model_pkl)
    model.load_state_dict(trained_state['model'])
    threshold = trained_state['threshold']
    return model, threshold


def main(positive_data, negative_data, dataset_name, graph_dict, side_effect_idx_dict, transR, transH, n_side_effect,
         d_hid, d_readout, n_attention_head, d_atom_feat, n_prop_step, n_atom_type, n_bond_type, best_model_pkl=None,
         batch_size=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("positive data", positive_data)
    print("negative data", negative_data)
    if dataset_name == "twosides":
        # create data loader
        test_data = prepare_ddi_testset_dataloader(
            positive_data, negative_data, batch_size, graph_dict, side_effect_idx_dict)
        print("Test Instance", test_data.__getitem__(0))
        print("batch_size", batch_size)
        # build model
        model, threshold = load_trained_model(transR, transH, device, n_side_effect, d_hid, d_readout, n_attention_head,
                                              n_prop_step,
                                              best_model_pkl, n_atom_type, n_bond_type, d_atom_feat)
        # print("Threshold", threshold)
        # start testing
        test_perf, _ = run_evaluation(model, test_data, device, dataset_name)
        for k, v in test_perf.items():
            if k != 'threshold':
                print(k, v)
            # print_performance_table({k: v for k, v in test_perf.items() if k != 'threshold'})
        return test_perf
