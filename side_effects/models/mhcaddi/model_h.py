import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import CoAttentionMessagePassingNetwork


class DrugDrugInteractionNetworkH(nn.Module):
	def __init__(
			self,
			n_atom_type, n_bond_type,
			d_node, d_edge, d_atom_feat, d_hid,
			n_prop_step,
			n_side_effect=None,
			n_lbls = 12,
			n_head=1, dropout=0.1,
			update_method='res', score_fn='trans'):

		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.atom_proj = nn.Linear(d_node + d_atom_feat, d_node)
		self.atom_emb = nn.Embedding(n_atom_type, d_node, padding_idx=0)
		self.bond_emb = nn.Embedding(n_bond_type, d_edge, padding_idx=0)
		nn.init.xavier_normal_(self.atom_emb.weight)
		nn.init.xavier_normal_(self.bond_emb.weight)

		self.side_effect_emb = None
		self.side_effect_norm_emb = None
		if n_side_effect is not None:
			self.side_effect_emb = nn.Embedding(n_side_effect, d_hid)
			self.side_effect_norm_emb = nn.Embedding(n_side_effect, d_hid)
			nn.init.xavier_normal_(self.side_effect_emb.weight)
			nn.init.xavier_normal_(self.side_effect_norm_emb.weight)

		self.encoder = CoAttentionMessagePassingNetwork(
			d_hid=d_hid, n_head=n_head, n_prop_step=n_prop_step,
			update_method=update_method, dropout=dropout)
		assert update_method == 'res'
		assert score_fn == 'trans'
		self.head_proj = nn.Linear(d_hid, d_hid, bias=False)
		self.tail_proj = nn.Linear(d_hid, d_hid, bias=False)
		nn.init.xavier_normal_(self.head_proj.weight)
		nn.init.xavier_normal_(self.tail_proj.weight)

		self.lbl_predict = nn.Linear(d_hid, n_lbls)

		self.__score_fn = score_fn


	@property
	def score_fn(self):
		return self.__score_fn

	def forward(
			self,
			seg_m1, atom_type1, atom_feat1, bond_type1,
			inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom_type2, atom_feat2, bond_type2,
			inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
			se_idx, drug_se_seg):

		atom1 = self.dropout(self.atom_comp(atom_feat1, atom_type1))
		atom2 = self.dropout(self.atom_comp(atom_feat2, atom_type2))

		bond1 = self.dropout(self.bond_emb(bond_type1))
		bond2 = self.dropout(self.bond_emb(bond_type2))

		d1_vec, d2_vec = self.encoder(
			seg_m1, atom1, bond1, inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom2, bond2, inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2)

		# TODO: what does this do? select pred for specific se?
		d1_vec = d1_vec.index_select(0, drug_se_seg)
		d2_vec = d2_vec.index_select(0, drug_se_seg)

		h_d1_vec = self.head_proj(d1_vec)
		h_d2_vec = self.head_proj(d2_vec)
		t_d1_vec = self.tail_proj(d1_vec)
		t_d2_vec = self.tail_proj(d2_vec)

		if self.side_effect_emb is not None:
			se_vec = self.dropout(self.side_effect_emb(se_idx))
			se_norm = self.dropout(self.side_effect_norm_emb(se_idx))

			h_d1_vec = self.transH_proj(h_d1_vec, se_norm)
			h_d2_vec = self.transH_proj(h_d2_vec, se_norm)
			t_d1_vec = self.transH_proj(t_d1_vec, se_norm)
			t_d2_vec = self.transH_proj(t_d2_vec, se_norm)

			e_vecs = [se_vec, d1_vec, d2_vec,
					  h_d1_vec, h_d2_vec, t_d1_vec, t_d2_vec]

			fwd_score = self.cal_translation_score(
				head=h_d1_vec,
				tail=t_d2_vec,
				rel=se_vec)
			bwd_score = self.cal_translation_score(
				head=h_d2_vec,
				tail=t_d1_vec,
				rel=se_vec)
			score = fwd_score + bwd_score

			o_loss = self.cal_orthogonal_loss(se_vec, se_norm)
			n_loss = sum([self.cal_vec_norm_loss(v) for v in e_vecs])

			#return score, o_loss + n_loss
			return score, o_loss + n_loss, se_idx, d1_vec, d2_vec
		else:
			pred1 = self.lbl_predict(d1_vec)
			pred2 = self.lbl_predict(d2_vec)
			return pred1,pred2, d1_vec, d2_vec


	def embed(self, seg_m1, atom_type1, atom_feat1, bond_type1,
			inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom_type2, atom_feat2, bond_type2,
			inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
			se_idx=None, drug_se_seg=None):
		atom1 = self.atom_comp(atom_feat1, atom_type1)
		atom2 = self.atom_comp(atom_feat2, atom_type2)
		bond1 = self.bond_emb(bond_type1)
		bond2 = self.bond_emb(bond_type2)

		d1_vec, d2_vec = self.encoder(
			seg_m1, atom1, bond1, inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom2, bond2, inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2)
		d1_vec = d1_vec.index_select(0, drug_se_seg)
		d2_vec = d2_vec.index_select(0, drug_se_seg)
		return d1_vec, d2_vec


	def transH_proj(self, original, norm):
		return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

	def atom_comp(self, atom_feat, atom_idx):
		atom_emb = self.atom_emb(atom_idx)
		node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
		return node

	def cal_translation_score(self, head, tail, rel):
		return torch.norm(head + rel - tail, dim=1)

	def cal_vec_norm_loss(self, vec, dim=1):
		norm = torch.norm(vec, dim=dim)
		return torch.mean(F.relu(norm - 1))

	def cal_orthogonal_loss(self, rel_emb, norm_emb):
		a = torch.sum(norm_emb * rel_emb, dim=1, keepdim=True) ** 2
		b = torch.sum(rel_emb ** 2, dim=1, keepdim=True) + 1e-6
		return torch.sum(a / b)
