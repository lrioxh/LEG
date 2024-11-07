
import logging
from torch import nn
from peft import LoraConfig, PeftModel, TaskType

from transformers import AutoConfig, AutoModel, DebertaV2Config, DebertaV2Model
from transformers import logging as transformers_logging
from src.model.lms.modules import DebertaClassificationHead, SentenceClsHead
from src.model.gnns.modules.GraphSAGE import SAGE

import copy
import torch
import torch.nn.functional as F
from dgl import function as fn
from dgl import DGLGraph
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn import SAGEConv

from src.misc.revgat.rev import memgcn
from src.misc.revgat.rev.rev_layer import SharedDropout

import src.lora as lora

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()

def _init_lora_linear(in_feats, out_feats, lora_params, **kwargs):
    if lora_params and lora_params.get('use_lora', False):
        return lora.Linear(
            in_feats, out_feats,
            lora_params['r'], lora_params['lora_alpha'], lora_params['lora_dropout'], **kwargs
        )
    else:
        return nn.Linear(in_feats, out_feats, **kwargs)

def _init_lora_emb(in_feats, out_feats, lora_params, **kwargs):
    if lora_params and lora_params.get('use_lora', False):
        return lora.Embedding(
            in_feats, out_feats,
            lora_params['r'], lora_params['lora_alpha'], lora_params['lora_dropout'], **kwargs
        )
    else:
        return nn.Embedding(in_feats, out_feats, **kwargs)

# LMs
class E5_model(nn.Module):
    def __init__(self, args):
        super(E5_model, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        logger.warning(f"inherit model weights from {pretrained_repo}")
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        config.save_pretrained(save_directory=args.save)
        # config['name_or_path'] = args.pretrained_dir
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config, add_pooling_layer=False)
        self.head = SentenceClsHead(config)
        if args.use_peft:
            lora_config = LoraConfig(   #TODO: 只微调前后层layers_to_transform 
                task_type=TaskType.SEQ_CLS, # .CAUSAL_LM
                inference_mode=False,
                r=args.peft_r_lm,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
                # layers_to_transform = [],
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()    # trainable params:

    def average_pool(self, last_hidden_states, attention_mask):  # for E5_model
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]#(100,512,1024)

    def forward(self, input_ids, att_mask, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.average_pool(bert_out.last_hidden_state, att_mask) #last_hidden_state as feat
        out = self.head(sentence_embeddings)

        if return_hidden:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return out, sentence_embeddings
        else:
            return out

class Deberta(nn.Module):
    def __init__(self, args):
        super(Deberta, self).__init__()
        transformers_logging.set_verbosity_error()
        pretrained_repo = args.pretrained_repo
        assert pretrained_repo in ["microsoft/deberta-v3-base"]
        config = AutoConfig.from_pretrained(pretrained_repo)
        config.num_labels = args.num_labels
        config.header_dropout_prob = args.header_dropout_prob
        if not args.use_default_config:
            config.hidden_dropout_prob = args.hidden_dropout_prob
            config.attention_probs_dropout_prob = args.attention_dropout_prob
        else:
            logger.warning("Using default config")
        config.save_pretrained(save_directory=args.save)
        # init modules
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, config=config)
        self.head = DebertaClassificationHead(config)
        if args.use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.peft_r_lm,
                lora_alpha=args.peft_lora_alpha,
                lora_dropout=args.peft_lora_dropout,
            )
            self.bert_model = PeftModel(self.bert_model, lora_config)
            self.bert_model.print_trainable_parameters()

    def forward(self, input_ids, att_mask, return_hidden=False):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        out = self.head(bert_out[0])
        if return_hidden:
            return out, bert_out[0][:, 0, :]
        else:
            return out

# GNNs

class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, use_gpt_preds: bool = False):
        super(GraphSAGE, self).__init__()
        
        self.dropout_rate = dropout
        self.use_gpt_preds = use_gpt_preds
        
        if use_gpt_preds:
            self.encoder = nn.Embedding(out_channels + 1, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels,aggregator_type='mean'))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels,aggregator_type='mean'))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels,aggregator_type='mean'))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_gpt_preds:
            self.encoder.reset_parameters()

    def forward(self, graph: DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "encoder"):
            feat = self.encoder(feat)
            feat = torch.flatten(feat, start_dim=1)
        for i, conv in enumerate(self.convs):
            feat = conv(graph, feat)
            if i < len(self.convs) - 1:
                feat = feat.relu_()
                feat = F.dropout(feat, p=self.dropout_rate, training=self.training)
        return feat

class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
        lora_params = None
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = _init_lora_linear(self._in_src_feats, out_feats * num_heads, lora_params, bias=False)
            self.fc_dst = _init_lora_linear(self._in_dst_feats, out_feats * num_heads, lora_params, bias=False)
        else:
            self.fc = _init_lora_linear(self._in_src_feats, out_feats * num_heads, lora_params, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0  # not implemented
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0  # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = _init_lora_linear(self._in_dst_feats, num_heads * out_feats, lora_params, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class RevGATBlock(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        edge_emb,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
        lora_params = None
    ):
        super(RevGATBlock, self).__init__()

        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = GATConv(
            node_feats,
            out_feats,
            num_heads=n_heads,
            attn_drop=attn_drop,
            edge_drop=edge_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            use_attn_dst=use_attn_dst,
            allow_zero_in_degree=allow_zero_in_degree,
            use_symmetric_norm=use_symmetric_norm,
            lora_params=lora_params
        )
        self.dropout = SharedDropout()
        if edge_emb > 0:
            self.edge_encoder = _init_lora_linear(edge_feats, edge_emb, lora_params)
        else:
            self.edge_encoder = None

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        if isinstance(self.dropout, SharedDropout):
            self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)

        if self.edge_encoder is not None:
            if efeat is None:
                efeat = graph.edata["feat"]
            efeat_emb = self.edge_encoder(efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)
        else:
            efeat_emb = None

        out = self.conv(graph, out, perm).flatten(1, -1)
        return out


class RevGAT(nn.Module):
    def __init__(
        self,
        args,
        activation,
        gpt_col = 5,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        group=2,
        use_gpt_preds=False,
        input_norm=True,
        lora_params = None
    ):
        super().__init__()
        self.in_feats = args.n_node_feats
        self.n_hidden = args.n_hidden
        self.n_classes = args.num_labels
        self.n_layers = args.n_layers
        self.num_heads = args.n_heads
        self.group = group
        self.gpt_col = gpt_col

        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(self.num_heads * self.n_hidden)
        if input_norm:
            self.input_norm = nn.BatchNorm1d(self.in_feats)

        if use_gpt_preds:
            self.encoder = _init_lora_emb(self.n_classes + 1, args.n_gpt_embs, lora_params)

        for i in range(self.n_layers):
            in_hidden = self.num_heads * self.n_hidden if i > 0 else self.in_feats
            out_hidden = self.n_hidden if i < self.n_layers - 1 else self.n_classes
            num_heads = self.num_heads if i < self.n_layers - 1 else 1
            out_channels = self.num_heads

            if i == 0 or i == self.n_layers - 1: 
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                        lora_params = lora_params
                    )
                )
            # elif i == self.n_layers - 1:
            #     self.convs.append(
            #         GATConv(
            #             in_hidden,
            #             out_hidden,
            #             num_heads=num_heads,
            #             attn_drop=attn_drop,
            #             edge_drop=edge_drop,
            #             use_attn_dst=use_attn_dst,
            #             use_symmetric_norm=use_symmetric_norm,
            #             residual=True,
            #         )
            #     )
            else:
                Fms = nn.ModuleList()
                fm = RevGATBlock(
                    in_hidden // group,
                    0,
                    0,
                    out_hidden // group,
                    n_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                    lora_params = lora_params
                )
                for i in range(self.group):
                    if i == 0:
                        Fms.append(fm)
                    else:
                        Fms.append(copy.deepcopy(fm))

                invertible_module = memgcn.GroupAdditiveCoupling(Fms, group=self.group)

                conv = memgcn.InvertibleModuleWrapper(fn=invertible_module, keep_input=False)

                self.convs.append(conv)

        self.bias_last = ElementWiseLinear(self.n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        x = feat
        if hasattr(self, "encoder"):
            embs = self.encoder(x[:, :self.gpt_col].to(torch.long))
            embs = torch.flatten(embs, start_dim=1)
            x = torch.cat([embs, x[:, self.gpt_col:]], dim=1)
        if hasattr(self, "input_norm"):
            x = self.input_norm(x)
        x = self.input_drop(x)

        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(), device=graph.device)
            self.perms.append(perm)

        x = self.convs[0](graph, x, self.perms[0]).flatten(1, -1)

        m = torch.zeros_like(x).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for i in range(1, self.n_layers - 1):
            graph.requires_grad = False
            perm = torch.stack([self.perms[i]] * self.group, dim=1)
            x = self.convs[i](x, graph, mask, perm)

        x = self.norm(x)
        x = self.activation(x, inplace=True)
        x = self.dp_last(x)
        x = self.convs[-1](graph, x, self.perms[-1])

        x = x.mean(1)
        x = self.bias_last(x)

        return x
