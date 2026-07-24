"""Checkpoint-compatible minimal TxGNN/RGCN model used by Stage 07.

The parameter names and forward interfaces intentionally match the legacy
``kg_txgnn_minimal.py``.  In particular, checkpoints remain plain
``HeteroRGCN.state_dict()`` mappings and are loaded with ``strict=True``.
"""

from __future__ import annotations

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


class DistMultPredictor(nn.Module):
    def __init__(
        self,
        n_hid,
        w_rels,
        G,
        rel2idx,
        proto=False,
        proto_num=3,
        sim_measure="all_nodes_profile",
        bert_measure="disease_name",
        agg_measure="learn",
        num_walks=200,
        walk_mode="bit",
        path_length=2,
        split="random",
        data_folder=None,
        exp_lambda=0.7,
        device="cpu",
    ):
        super().__init__()
        if proto:
            raise NotImplementedError(
                "The self-contained KG workflow supports prototype_learning=false, "
                "which is the checkpoint-compatible production setting."
            )
        self.proto = proto
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx
        self.etypes_dd = [
            ("drug", "contraindication", "disease"),
            ("drug", "indication", "disease"),
            ("drug", "off-label use", "disease"),
            ("disease", "rev_contraindication", "drug"),
            ("disease", "rev_indication", "drug"),
            ("disease", "rev_off-label use", "drug"),
        ]
        self.node_types_dd = ["disease", "drug"]

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        h_r = self.W[self.rel2idx[edges._etype]]
        return {"score": torch.sum(h_u * h_r * h_v, dim=1)}

    def forward(self, graph, G, h, pretrain_mode, mode, block=None, only_relation=None):
        with graph.local_scope():
            scores = {}
            score_list = []
            etypes_train = (
                graph.canonical_etypes if len(graph.canonical_etypes) == 1 else self.etypes_dd
            )
            if only_relation is not None:
                mapping = {
                    "indication": [
                        ("drug", "indication", "disease"),
                        ("disease", "rev_indication", "drug"),
                    ],
                    "contraindication": [
                        ("drug", "contraindication", "disease"),
                        ("disease", "rev_contraindication", "drug"),
                    ],
                    "off-label": [
                        ("drug", "off-label use", "disease"),
                        ("disease", "rev_off-label use", "drug"),
                    ],
                }
                if only_relation not in mapping:
                    raise ValueError(f"Unsupported only_relation: {only_relation}")
                etypes_train = mapping[only_relation]
            graph.ndata["h"] = h
            if pretrain_mode:
                etypes_train = [
                    etype for etype in graph.canonical_etypes if graph.num_edges(etype=etype) != 0
                ]
            for etype in etypes_train:
                if etype not in graph.canonical_etypes or graph.num_edges(etype=etype) == 0:
                    continue
                graph.apply_edges(self.apply_edges, etype=etype)
                out = graph.edges[etype].data["score"]
                if pretrain_mode:
                    out = torch.sigmoid(out)
                scores[etype] = out
                score_list.append(out)
            if not score_list:
                empty = torch.empty(0, device=self.W.device)
                return scores, empty if pretrain_mode else empty.detach().cpu().numpy()
            out_all = torch.cat(score_list)
            if not pretrain_mode:
                out_all = out_all.reshape(-1).detach().cpu().numpy()
            return scores, out_all


class AttHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super().__init__()
        self.weight = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
        self.attn_fc = nn.ModuleDict(
            {name: nn.Linear(out_size * 2, 1, bias=False) for name in etypes}
        )

    def edge_attention(self, edges):
        srctype, etype, dsttype = edges._etype
        if srctype == dsttype:
            wh2 = torch.cat([edges.src[f"Wh_{etype}"], edges.dst[f"Wh_{etype}"]], dim=1)
        elif etype[:3] == "rev":
            wh2 = torch.cat([edges.src[f"Wh_{etype}"], edges.dst[f"Wh_{etype[4:]}"]], dim=1)
        else:
            wh2 = torch.cat([edges.src[f"Wh_{etype}"], edges.dst[f"Wh_rev_{etype}"]], dim=1)
        return {f"e_{etype}": F.leaky_relu(self.attn_fc[etype](wh2))}

    def message_func(self, edges):
        etype = edges._etype[1]
        return {"m": edges.src[f"Wh_{etype}"], "e": edges.data[f"e_{etype}"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        return {"h": torch.sum(alpha * nodes.mailbox["m"], dim=1)}

    def forward(self, G, feat_dict, return_att=False):
        with G.local_scope():
            funcs = {}
            att = {}
            etypes_all = [e for e in G.canonical_etypes if G.num_edges(etype=e) != 0]
            for srctype, etype, _ in etypes_all:
                G.nodes[srctype].data[f"Wh_{etype}"] = self.weight[etype](feat_dict[srctype])
            for srctype, etype, dsttype in etypes_all:
                canonical = (srctype, etype, dsttype)
                G.apply_edges(self.edge_attention, etype=canonical)
                if return_att:
                    att[canonical] = G.edges[etype].data[f"e_{etype}"].detach().cpu().numpy()
                funcs[etype] = (self.message_func, self.reduce_func)
            G.multi_update_all(funcs, "sum")
            return {n: G.dstdata["h"][n] for n in G.dstdata["h"].keys()}, att


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super().__init__()
        self.weight = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
        self.in_size = in_size
        self.out_size = out_size
        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, _ in [e for e in G.canonical_etypes if G.num_edges(etype=e)]:
            G.nodes[srctype].data[f"Wh_{etype}"] = self.weight[etype](feat_dict[srctype])
            funcs[etype] = (fn.copy_u(f"Wh_{etype}", "m"), fn.mean("m", "h"))
        G.multi_update_all(funcs, "sum")
        return {n: G.dstdata["h"][n] for n in G.dstdata["h"].keys()}

    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate, self.baseline, self.layer = gate, baseline, layer

    def graphmask_forward(self, *args, **kwargs):
        raise NotImplementedError("GraphMask is not used by the KG workflow.")


class HeteroRGCN(nn.Module):
    def __init__(
        self,
        G,
        in_size,
        hidden_size,
        out_size,
        attention=False,
        proto=False,
        proto_num=3,
        sim_measure="all_nodes_profile",
        bert_measure="disease_name",
        agg_measure="learn",
        num_walks=200,
        walk_mode="bit",
        path_length=2,
        split="random",
        data_folder=None,
        exp_lambda=0.7,
        device="cpu",
    ):
        super().__init__()
        layer = AttHeteroRGCNLayer if attention else HeteroRGCNLayer
        self.layer1 = layer(in_size, hidden_size, G.etypes)
        self.layer2 = layer(hidden_size, out_size, G.etypes)
        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain("relu"))
        rel2idx = dict(zip(G.canonical_etypes, range(len(G.canonical_etypes))))
        self.pred = DistMultPredictor(
            hidden_size,
            self.w_rels,
            G,
            rel2idx,
            proto=proto,
            proto_num=proto_num,
            sim_measure=sim_measure,
            bert_measure=bert_measure,
            agg_measure=agg_measure,
            num_walks=num_walks,
            walk_mode=walk_mode,
            path_length=path_length,
            split=split,
            data_folder=data_folder,
            exp_lambda=exp_lambda,
            device=device,
        )
        self.attention = attention
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes
        self.device = device

    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode="train", pretrain_mode=False):
        input_dict = blocks[0].srcdata["inp"]
        h_dict = self.layer1(blocks[0], input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(blocks[1], h_dict)
        scores, out_pos = self.pred(pos_G, G, h, pretrain_mode, mode=mode + "_pos", block=blocks[1])
        scores_neg, out_neg = self.pred(
            neg_G, G, h, pretrain_mode, mode=mode + "_neg", block=blocks[1]
        )
        return scores, scores_neg, out_pos, out_neg

    def forward(
        self,
        G,
        neg_G,
        eval_pos_G=None,
        return_h=False,
        return_att=False,
        mode="train",
        pretrain_mode=False,
    ):
        with G.local_scope():
            input_dict = {ntype: G.nodes[ntype].data["inp"] for ntype in G.ntypes}
            if self.attention:
                h_dict, a1 = self.layer1(G, input_dict, return_att)
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
                h, a2 = self.layer2(G, h_dict, return_att)
            else:
                h_dict = self.layer1(G, input_dict)
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
                h = self.layer2(G, h_dict)
            if return_h:
                return h
            if return_att:
                return a1, a2
            positive_graph = eval_pos_G if eval_pos_G is not None else G
            scores, out_pos = self.pred(positive_graph, G, h, pretrain_mode, mode=mode + "_pos")
            scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode=mode + "_neg")
            return scores, scores_neg, out_pos, out_neg

    def graphmask_forward(self, *args, **kwargs):
        raise NotImplementedError("GraphMask is not used by the KG workflow.")

    def enable_layer(self, layer, graphmask=True):
        return None

    def count_layers(self):
        return 2

    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]

    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]

    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]

    def add_graphmask_parameters(self, *args, **kwargs):
        raise NotImplementedError("GraphMask is not used by the KG workflow.")


def construct_negative_graph_each_etype(graph, k, etype, method, weights, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    if method == "corrupt_dst":
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    elif method == "corrupt_src":
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph.num_nodes(utype), (len(dst) * k,))
    elif method == "corrupt_both":
        neg_src = torch.randint(0, graph.num_nodes(utype), (len(dst) * k,))
        neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    elif method in {"multinomial_src", "inverse_src", "fix_src"}:
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except RuntimeError:
            neg_src = torch.tensor([], dtype=torch.int64)
    elif method in {"multinomial_dst", "inverse_dst", "fix_dst"}:
        neg_src = src.repeat_interleave(k)
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True)
        except RuntimeError:
            neg_dst = torch.tensor([], dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported negative sampling method: {method}")
    return {etype: (neg_src.to(device), neg_dst.to(device))}


class Full_Graph_NegSampler:
    def __init__(self, g, k, method, device):
        if method == "multinomial_src":
            self.weights = {e: g.out_degrees(etype=e).float() ** 0.75 for e in g.canonical_etypes}
        elif method == "multinomial_dst":
            self.weights = {e: g.in_degrees(etype=e).float() ** 0.75 for e in g.canonical_etypes}
        elif method == "inverse_dst":
            self.weights = {e: -(g.in_degrees(etype=e).float() ** 0.75) for e in g.canonical_etypes}
        elif method == "inverse_src":
            self.weights = {
                e: -(g.out_degrees(etype=e).float() ** 0.75) for e in g.canonical_etypes
            }
        elif method == "fix_dst":
            self.weights = {e: (g.in_degrees(etype=e) > 0).float() for e in g.canonical_etypes}
        elif method == "fix_src":
            self.weights = {e: (g.out_degrees(etype=e) > 0).float() for e in g.canonical_etypes}
        else:
            self.weights = {}
        self.k, self.method, self.device = k, method, device

    def __call__(self, graph):
        out = {}
        for etype in graph.canonical_etypes:
            temp = construct_negative_graph_each_etype(
                graph, self.k, etype, self.method, self.weights, self.device
            )
            if len(temp[etype][0]):
                out.update(temp)
        return dgl.heterograph(
            out,
            num_nodes_dict={n: graph.num_nodes(n) for n in graph.ntypes},
        )


class Minibatch_NegSampler:
    def __init__(self, g, k, method):
        if method == "multinomial_dst":
            self.weights = {e: g.in_degrees(etype=e).float() ** 0.75 for e in g.canonical_etypes}
        elif method == "fix_dst":
            self.weights = {e: (g.in_degrees(etype=e) > 0).float() for e in g.canonical_etypes}
        elif method == "multinomial_src":
            self.weights = {e: g.out_degrees(etype=e).float() ** 0.75 for e in g.canonical_etypes}
        elif method == "fix_src":
            self.weights = {e: (g.out_degrees(etype=e) > 0).float() for e in g.canonical_etypes}
        else:
            self.weights = {}
        self.k, self.method = k, method

    def __call__(self, g, eids_dict):
        result = {}
        for etype, eids in eids_dict.items():
            src, dst = g.find_edges(eids, etype=etype)
            if self.method in {"multinomial_src", "fix_src"}:
                neg_dst = dst.repeat_interleave(self.k)
                neg_src = self.weights[etype].multinomial(len(neg_dst), replacement=True)
            else:
                neg_src = src.repeat_interleave(self.k)
                neg_dst = self.weights[etype].multinomial(len(neg_src), replacement=True)
            result[etype] = (neg_src, neg_dst)
        return result


def get_all_metrics_fb(pred_pos, pred_neg, scores, labels, G, full_mode=False):
    auroc_rel, auprc_rel = {}, {}
    etypes = G.canonical_etypes if full_mode else []
    for etype in etypes:
        try:
            pos = pred_pos[etype].reshape(-1).detach().cpu().numpy()
            neg = pred_neg[etype].reshape(-1).detach().cpu().numpy()
            prediction = np.concatenate((pos, neg))
            truth = [1] * len(pos) + [0] * len(neg)
            auroc_rel[etype] = roc_auc_score(truth, prediction)
            auprc_rel[etype] = average_precision_score(truth, prediction)
        except Exception:
            pass
    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = float(np.mean(list(auroc_rel.values()))) if auroc_rel else np.nan
    macro_auprc = float(np.mean(list(auprc_rel.values()))) if auprc_rel else np.nan
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc


def evaluate_graph_construct(df_valid, g, neg_sampler, k, device):
    out = {}
    for etype in g.canonical_etypes:
        temp = df_valid[df_valid.relation == etype[1]]
        out[etype] = (
            torch.as_tensor(temp.x_idx.values, dtype=torch.int64, device=device),
            torch.as_tensor(temp.y_idx.values, dtype=torch.int64, device=device),
        )
    positive = dgl.heterograph(out, num_nodes_dict={n: g.num_nodes(n) for n in g.ntypes})
    negative = Full_Graph_NegSampler(positive, k, neg_sampler, device)(positive)
    return positive, negative


def create_dgl_graph(df_train, df):
    dgl_input = {}
    for x_type, relation, y_type in (
        df_train[["x_type", "relation", "y_type"]].drop_duplicates().values
    ):
        edges = df_train[df_train.relation == relation][["x_idx", "y_idx"]].values.T
        dgl_input[(x_type, relation, y_type)] = (
            edges[0].astype(int),
            edges[1].astype(int),
        )
    maxima = {}
    for column_type, column_index in (("x_type", "x_idx"), ("y_type", "y_idx")):
        for key, value in df.groupby(column_type)[column_index].max().items():
            maxima[key] = max(maxima.get(key, -1), value)
    maxima.setdefault("effect/phenotype", 0)
    graph = dgl.heterograph(
        dgl_input, num_nodes_dict={key: int(value) + 1 for key, value in maxima.items()}
    )
    for edge_id, etype in enumerate(graph.etypes):
        graph.edges[etype].data["id"] = torch.full(
            (graph.num_edges(etype),), edge_id, dtype=torch.long
        )
    return graph


def initialize_node_embedding(g, n_inp):
    for ntype in g.ntypes:
        emb = nn.Parameter(torch.empty(g.num_nodes(ntype), n_inp), requires_grad=False)
        nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data["inp"] = emb
    return g


__all__ = [
    "Full_Graph_NegSampler",
    "HeteroRGCN",
    "Minibatch_NegSampler",
    "create_dgl_graph",
    "evaluate_graph_construct",
    "get_all_metrics_fb",
    "initialize_node_embedding",
]
