import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionLayer
from .features_extraction import FeaturesExtractorFactory


class SelfAttentionLayer(AttentionLayer):
    """
    A self-attention block. The only difference is that the key,
    query and value tensors are one and the same.
    """

    def __init__(self, key_dim, pooling_function=None):
        super().__init__(1, 1, key_dim, pooling_function)
        # self.query_network = self.key_network
        # self.value_network = nn.Sequential()

    def forward(self, x):
        """
        Computes the self-attention.

        Parameters
        ----------
        x: torch.Tensor
            This is self attention, which means that x is used as the key, query and value tensors.
        Returns
        -------
        attention: torch.Tensor
            The attention block.
        """

        x = x.reshape(*x.shape, 1)
        return super().forward(x, x, x).squeeze(-1)


class BMNDDI(nn.Module):
    def __init__(self, drug_feature_extractor_params, fc_layers_dim, nb_side_effects=1, mode='concat',
                 att_mode=None, ss_embedding_dim=None, add_feats_params=None, is_binary_output=False, testing=False,
                 exp_prefix=None,
                 is_multitask_output=False, on_multi_dataset=False, op_mode=None, pretrained_feature_extractor=False,
                 trainer=None, **kwargs):

        super(BMNDDI, self).__init__()
        fe_factory = FeaturesExtractorFactory()
        self.mode = mode
        self.train_fn = trainer
        self.att_mode = att_mode
        self.is_binary_output = is_binary_output
        self.testing = testing
        self.exp_prefix = exp_prefix
        self.is_multitask_output = is_multitask_output
        self.on_multidataset = on_multi_dataset
        self.op_mode = op_mode
        self.use_basic_model = (not is_binary_output) and (not is_multitask_output)
        self.trainer = trainer

        # This the place that need to be update
        if pretrained_feature_extractor:
            self.drug_feature_extractor = self.load_pretrained_features_extractor(**drug_feature_extractor_params)

        else:
            self.drug_feature_extractor = fe_factory(**drug_feature_extractor_params)

        in_size = 2 * self.drug_feature_extractor.output_dim

        if self.mode in ['sum', 'max', "elementwise"]:
            in_size = self.drug_feature_extractor.output_dim

        if is_binary_output:
            self.embedding = nn.Embedding(nb_side_effects, ss_embedding_dim)
            in_size += ss_embedding_dim

        if add_feats_params:
            self.add_feature_extractor = fe_factory(arch='fcnet', last_layer_activation=None, **add_feats_params,
                                                    **kwargs)
            in_size += add_feats_params.pop("nb_side_effects")

        output_dim = 1 if is_binary_output else nb_side_effects

        # This part is not so clean ---
        # ##TOD -- make the last_layer more suitble for the task type

        if output_dim == 1 and (not is_binary_output):
            last_layer = None
        else:
            last_layer = "Sigmoid"

        self.classifier = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                     output_dim=output_dim, last_layer_activation=last_layer, **kwargs)

        if self.is_multitask_output and op_mode is None:  # chabge op for task param; need to be set all the auxiliary task layer ouptus
            self.dist_fc = fe_factory(arch='fcnet', input_size=in_size, fc_layer_dims=fc_layers_dim,
                                      output_dim=1, last_layer_activation=None, **kwargs)

    def forward(self, batch):

        add_feats = None
        similarity = None
        # print("ok")
        if self.is_binary_output:
            # print("1")
            if self.testing:
                did1, did2, sid, drugs_a, drugs_b, side_eff = batch[:6]
                add_feats = batch[6:]
                if isinstance(side_eff, tuple):
                    side_eff = torch.cat([s.reshape(1) for s in side_eff])
            else:
                drugs_a, drugs_b, side_eff = batch[:3]
                add_feats = batch[3:]
                if isinstance(side_eff, tuple):
                    side_eff = torch.cat(side_eff)
            side_eff_features = self.embedding(side_eff).squeeze()

        # this part wil be removed soon
        if self.is_multitask_output and self.on_multidataset:
            # print("2") -- need to be updated since the data loader was removed
            task_a, task_b = batch
            drugs_a, drugs_b = task_a
            drugs_c, drugs_d = task_b
            drugs_a, drugs_b, drugs_c, drugs_d = drugs_a.view(-1, drugs_a.size(2)), drugs_b.view(-1, drugs_b.size(
                2)), drugs_c.view(-1, drugs_c.size(2)), drugs_d.view(-1, drugs_d.size(2))
            # print(drugs_a.shape, drugs_b.shape, drugs_c.shape, drugs_d.shape)
            features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
            # print(features_drug1.shape, features_drug2.shape)
            features_drug3, features_drug4 = self.drug_feature_extractor(drugs_c), self.drug_feature_extractor(drugs_d)
            # print(features_drug3.shape, features_drug4.shape)
            ddi_feats = torch.cat((features_drug1, features_drug2), 1)
            op_feats = torch.cat((features_drug3, features_drug4), 1)
            # print(ddi_feats.shape)
            ddi = self.classifier(ddi_feats)
            sim = self.fusion_layer(features_drug3, features_drug4,
                                    mode=self.op_mode) if self.op_mode else self.dist_fc(op_feats)
            return ddi, sim

        if self.use_basic_model:
            return self.bmn_basic(batch)

        if self.is_multitask_output and not self.on_multidataset:
            drugs_a, drugs_b = batch
            # print("4", drugs_a.shape, drugs_b.shape)
            features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
            feats = torch.cat((features_drug1, features_drug2), 1)
            ddi = self.classifier(feats)
            sim = self.fusion_layer(features_drug1, features_drug2,
                                    mode=self.op_mode) if self.op_mode else self.dist_fc(feats)
            return ddi, sim

    def set_graph(self, nodes, edges):
        self.nb_nodes = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.adj_mat = (edges.sum(2) > 0).float()

    def fusion_layer(self, vec1, vec2, mode="cc"):
        if mode == 'cos':
            vec = F.cosine_similarity(
                vec1 + 1e-16, vec2 + 1e-16, dim=-1)
            vec = vec.unsqueeze(1)
        elif mode == 'l1':
            vec = self.dist_fc(torch.abs(vec1 - vec2))
            vec = vec.squeeze(1)
        elif mode == 'l2':
            vec = self.dist_fc(torch.abs(vec1 - vec2) ** 2)
            vec = vec.squeeze(1)
        elif mode == "elementwise":
            vec = torch.mul(vec1, vec2)
        elif mode == "sum":
            vec = torch.add(vec1, vec2)
        elif mode == "max":
            vec = torch.max(vec1, vec2)
        else:
            vec = torch.cat((vec1, vec2), 1)

        return vec

    def load_pretrained_features_extractor(self, task_id, requires_grad):
        config = json.load(open(f"{task_id}_params.json"))
        model_params = config["model_params"]
        model = self.trainer(network_params=model_params["network_params"])
        checkpoint = torch.load(f"{task_id}.checkpoint.pth", map_location="cpu")
        model.model.load_state_dict(checkpoint['net'])

        if not requires_grad:
            for param in model.model.parameters():
                param.requires_grad = False
        return model.model.drug_feature_extractor

    def bmn_basic(self, batch):
      #  side_eff_features = None
        drugs_a, drugs_b, = batch[:2]
        add_feats = batch[2:]
        features_drug1, features_drug2 = self.drug_feature_extractor(drugs_a), self.drug_feature_extractor(drugs_b)
        ddi = self.fusion_layer(features_drug1, features_drug2, mode=self.mode)
        if add_feats:
            add_feats = self.add_feature_extractor(torch.cat(add_feats, 1))
            ddi = torch.cat((ddi, add_feats), 1)
        #
        # if side_eff_features is not None:
        #     ddi = torch.cat((ddi, side_eff_features), 1)
        side_effect = self.classifier(ddi)
        return side_effect
