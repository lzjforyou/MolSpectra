import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model import load_pretrained

from misc_utils import DummyContext, th_temp_seed
from MolSpectralModel import MolSpectraEmbedder

def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx) # Ensure all precursor mass indices prec_mass_idx are less than max_idx, otherwise index out of bounds
    idx = th.arange(max_idx, device=prec_mass_idx.device) # Create a 1D tensor idx of size [D], representing the spectrum index range
    mask = (
        idx.unsqueeze(0) <= (
            prec_mass_idx.unsqueeze(1) +
            prec_mass_offset)).float() # Compare whether idx is less than or equal to the above range, generating a [B, D] mask matrix with elements 0 or 1
    return mask * raw_prediction + (1. - mask) * mask_value # mask * raw_prediction retains prediction values within range. (1. - mask) * mask_value replaces values outside range with mask_value


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]
    # raw_prediction: Original prediction values, shape [B, D], where B is batch size and D is spectrum dimension
    # prec_mass_idx: Mass index of precursor ion, shape [B], representing the precursor mass corresponding to each sample
    # prec_mass_offset: Mass offset, used to adjust the reference point for reverse prediction

    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    rev_prediction = th.flip(raw_prediction, dims=(1,)) # Flip raw_prediction along dimension 1 (spectrum dimension) to generate reverse prediction rev_prediction
    # convention is to shift right, so we express as negative to go left
    offset_idx = th.minimum(
        max_idx * th.ones_like(prec_mass_idx),
        prec_mass_idx + prec_mass_offset + 1) # Add offset and 1 to precursor mass index, limit the offset index within spectrum dimension range to ensure valid index
    shifts = - (max_idx - offset_idx)
    gather_idx = th.arange( # Generate index sequence from 0 to max_idx-1, expand batch dimension of index matrix to shape [B, D]
        max_idx,
        device=raw_prediction.device).unsqueeze(0).expand(
        batch_size,
        max_idx)
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx # Adjust index matrix according to offset. Use modulo operation to limit all indices within valid range [0, max_idx)
    offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx) # Use gather_idx to retrieve values from rev_prediction, generating adjusted reverse prediction offset_rev_prediction
    # you could mask_prediction_by_mass here but it's unnecessary 
    return offset_rev_prediction


def test_mask_prediction():

    x = th.arange(3 * 3).reshape(3, 3).float()
    mask_idx = th.arange(3)
    x_mask_0 = th.tensor([[0., 0., 0.], [3., 4., 0.], [6., 7., 8.]])
    x_mask_1 = th.tensor([[0., 1., 0.], [3., 4., 5.], [6., 7., 8.]])
    print(x)
    print(x_mask_0)
    print(mask_prediction_by_mass(x, mask_idx, 0))
    print(x_mask_1)
    print(mask_prediction_by_mass(x, mask_idx, 1))
    print()


def test_reverse_prediction():

    x = th.arange(3 * 3).reshape(3, 3).float()
    shift_idx = th.arange(3)
    x_rev_shift_0 = th.tensor([[0., 2., 1.], [4., 3., 5.], [8., 7., 6.]])
    x_rev_shift_1 = th.tensor([[1., 0., 2.], [5., 4., 3.], [8., 7., 6.]])
    print(x)
    print(x_rev_shift_0)
    print(reverse_prediction(x, shift_idx, 0))
    print(x_rev_shift_1)
    print(reverse_prediction(x, shift_idx, 1))
    print()


"""
model takes in a batched Data object that has node_feats, edge_index, global_feats, spec

outputs the predicted spectrum

> best setup:
- 10 layers
- 64-dim hidden dim (throughout)
- linear embedding
- GLU to predict output
- the paper does not specify information about the FF NN (like skip connections, dimension, etc)
- LeakyReLU is used throughout since the GAT uses LeakyReLU
- num_heads was not specified (assuming 10)
- dropout was not specific (assuming 0.)
"""


class LinearBlock(nn.Module):

    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.bn(self.dropout(F.relu(self.linear(x))))


class NeimsBlock(nn.Module):
    """ from the NEIMS paper (uses LeakyReLU instead of ReLU) """

    def __init__(self, in_dim, out_dim, dropout):

        super(NeimsBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h


class Conv1dBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            pool_type="avg",
            pool_size=4,
            batch_norm=True,
            dropout=0.1,
            activation="relu"):
        """
        padding is always the same
        order of ops:
        1. conv
        2. batch norm (if applicable)
        3. activation
        4. pool (if applicable)
        5. dropout (if applicable)
        """

        super(Conv1dBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ops_list = []
        assert kernel_size % 2 == 1, kernel_size
        padding = kernel_size // 2
        ops_list.append(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding))
        if batch_norm:
            ops_list.append(nn.BatchNorm1d(out_channels))
        if activation == "relu":
            ops_list.append(nn.ReLU())
        else:
            raise ValueError
        if pool_type == "max":
            ops_list.append(nn.MaxPool1d(pool_size, ceil_mode=True))
        elif pool_type == "avg":
            ops_list.append(nn.AvgPool1d(pool_size, ceil_mode=True))
        else:
            assert pool_type == "none", pool_type
        if dropout > 0.:
            ops_list.append(nn.Dropout(dropout))
        self.ops = nn.Sequential(*ops_list)

    def forward(self, b_x):

        assert b_x.ndim == 3
        assert b_x.shape[1] == self.in_channels
        b_y = self.ops(b_x)
        return b_y






class Predictor(nn.Module):

    def __init__(self, dim_d, **kwargs):

        super(Predictor, self).__init__()
        self.g_dim = dim_d["g_dim"] # Dimension of metadata
        # self.g_dim = 0
        self.o_dim = dim_d["o_dim"] # Dimension of spectrum features
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.embedders = nn.ModuleList([])
        if "hdse" in self.embed_types:
            self.embedders.append(MolSpectralEmbedder(**kwargs))
        assert len(self.embedders) > 0, len(self.embedders)
        embeds_dims = [embedder.get_embed_dim() for embedder in self.embedders] # encoder_embed_dim Traverse all embedders, call their get_embed_dim method to get embedding vector dimension
        if self.g_dim > 0:
            embeds_dims.append(self.g_dim)
        if self.embed_dim == -1:
            # infer embed_dim Infer total embedding dimension after adding metadata
            self.embed_dim = sum(embeds_dims)
        if self.embed_linear: # If True, perform linear transformation on each embedding feature to unify them to total embedding dimension self.embed_dim
            self.embed_layers = nn.ModuleList(
                [nn.Linear(embed_dim, self.embed_dim) for embed_dim in embeds_dims])
        else:
            self.embed_layers = nn.ModuleList(
                [nn.Identity() for embed_dim in embeds_dims]) # If self.embed_linear is False, directly retain embedding features without any transformation
        self.ff_layers = nn.ModuleList([]) # Store multiple feedforward network layers
        self.out_modules = [] # Store module names
        if self.ff_layer_type == "standard":
            ff_layer = LinearBlock
        else:
            assert self.ff_layer_type == "neims", self.ff_layer_type
            ff_layer = NeimsBlock
        self.ff_layers.append(nn.Linear(self.embed_dim, self.ff_h_dim)) # Add first fully connected layer
        self.out_modules.extend(["ff_layers"]) # Add ff_layers module name to track module composition
        for i in range(self.ff_num_layers):
            self.ff_layers.append(
                ff_layer(
                    self.ff_h_dim,
                    self.ff_h_dim,
                    self.dropout))
        if self.bidirectional_prediction: # If enabled, model performs bidirectional prediction, i.e., processes both forward and reverse spectra simultaneously
           
            self.forw_out_layer = nn.Linear(self.ff_h_dim, self.o_dim) # Fully connected layer for forward spectrum prediction, mapping hidden dimension (ff_h_dim) to spectrum output dimension (o_dim)
            self.rev_out_layer = nn.Linear(self.ff_h_dim, self.o_dim) # Fully connected layer for reverse spectrum prediction, similar to forward prediction
            self.out_gate = nn.Sequential( # Use linear layer and Sigmoid activation function to implement gating mechanism, controlling weighted combination of forward and reverse spectrum outputs
                *[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
            self.out_modules.extend(
                ["forw_out_layer", "rev_out_layer", "out_gate"])
        else:
            self.out_layer = nn.Linear(self.ff_h_dim, self.o_dim) # For unidirectional prediction, only one output layer out_layer
            if self.gate_prediction:
                self.out_gate = nn.Sequential(
                    *[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
            self.out_modules.extend(["out_layer", "out_gate"])

    def forward(self, data, perturb=None, amp=False, return_input_feats="", return_lda_pred=False):

        if amp: # amp: Boolean variable indicating whether to enable Automatic Mixed Precision (AMP)
            amp_context = th.cuda.amp.autocast()
        else:
            amp_context = DummyContext()
        with amp_context:
            
            embeds = [embedder(data, perturb=perturb) for embedder in self.embedders]
            # add on metadata
            if self.g_dim > 0:
                embeds.append(data["spec_meta"])
            # apply transformation
            embeds = [self.embed_layers[embed_idx](embed) for embed_idx,embed in enumerate(embeds)]
            # aggregate
            if self.embed_agg == "concat":
                fh = th.cat(embeds, dim=1)
            elif self.embed_agg == "add":
                assert all(embed.shape[1] == embeds[0].shape[1]
                           for embed in embeds)
                fh = sum(embeds)
            elif self.embed_agg == "avg":
                fh = sum(embeds) / len(embeds)
            else:
                raise ValueError("invalid agg_embed")
            # apply feedforward layers
            fh = self.ff_layers[0](fh)
            for ff_layer in self.ff_layers[1:]:
                if self.ff_skip:
                    fh = fh + ff_layer(fh)
                else:
                    fh = ff_layer(fh)
            if self.bidirectional_prediction:
                ff = self.forw_out_layer(fh)
                fr = reverse_prediction(self.rev_out_layer(fh),data["prec_mz_idx"],self.prec_mass_offset) # prec_mass_offset: Offset indicating additional range to retain based on precursor mass index
                fg = self.out_gate(fh)
                fo = ff * fg + fr * (1. - fg)
                # Perform mask operation on spectrum prediction values, only retaining values within range of precursor mass-to-charge ratio index prec_mass_idx and its offset prec_mass_offset, setting values outside range to mask_value (default 0.0)
                fo = mask_prediction_by_mass(fo, data["prec_mz_idx"], self.prec_mass_offset)
            else:
                # apply output layer
                fo = self.out_layer(fh)
                # apply gating
                if self.gate_prediction:
                    fg = self.out_gate(fh)
                    fo = fg * fo
            # apply output activation
            if self.output_activation == "relu":
                output_activation_fn = F.relu
            elif self.output_activation == "sp":
                output_activation_fn = F.softplus
            elif self.output_activation == "sm":
                # you shouldn't gate with sm
                assert not self.bidirectional_prediction
                assert not self.gate_prediction
                assert not self.spectrum_attention
                def output_activation_fn(x): return F.softmax(x, dim=1)
            else:
                raise ValueError(f"invalid output_activation: {self.output_activation}")
            fo = output_activation_fn(fo)
            # apply gt gating Apply Ground Truth gating mechanism
            if self.gt_gate_prediction:
                # binarize gt spec
                gt_fo = (data["spec"] > 0.).float() # Set values greater than 0 in spectrum data to 1, others to 0
                # map binary to [1-gt_gate_val,gt_gate_val]
                assert self.gt_gate_val > 0.5
                gt_fo = gt_fo * (2 * self.gt_gate_val - 1.) + (1. - self.gt_gate_val) # For example, if gt_gate_val=0.8, non-zero values map to 0.8, zero values map to 0.2
                # multiply
                fo = gt_fo * fo # Multiply gt_fo with prediction fo, adjusting values at corresponding positions by weighting
            # apply spectrum attention
            if self.spectrum_attention:
                fo = self.spectrum_attender(fo)
                fo = output_activation_fn(fo)
            # apply normalization
            if self.output_normalization == "l1":
                fo = F.normalize(fo, p=1, dim=1)
            elif self.output_normalization == "l2":
                fo = F.normalize(fo, p=2, dim=1)
            elif self.output_normalization == "none":
                pass
            else:
                raise ValueError(
                    f"invalid output_normalization: {self.output_normalization}")
            output_d = {"pred":fo}
            return output_d


    def get_split_params(self):
        # nopt_params: Used to store non-pretrained parameters
        # pt_params: Used to store pretrained parameters
        nopt_params, pt_params = [], []
        for embedder in self.embedders:
            emb_nopt_params, emb_pt_params = embedder.get_split_params()
            nopt_params.extend(emb_nopt_params)
            pt_params.extend(emb_pt_params)
        for out_module in self.out_modules:
            nopt_params.extend(list(getattr(self, out_module).parameters()))
        return nopt_params, pt_params

    def count_parameters(self):

        # molecule embedder
        mol_params = 0
        for embedder in self.embedders:
            mol_params += sum(p.numel() for p in embedder.parameters())
        for embed_layer in self.embed_layers:
            mol_params += sum(p.numel() for p in embed_layer.parameters())
        # mlp
        mlp_params = 0
        for out_module in self.out_modules:
            mlp_params += sum(p.numel() for p in getattr(self, out_module).parameters())
        # total
        total_params = sum(p.numel() for p in self.parameters())
        return mol_params, mlp_params, total_params

    def set_mode(self, mode):
        pass


def dirichlet_sample(alpha):

    dist = th.distributions.dirichlet.Dirichlet(alpha)
    return dist.sample()


class SpectrumAttention(nn.Module):

    def __init__(self, D, H, N, alpha=10.):
        super().__init__()
        self.pi = nn.Parameter(
            data=dirichlet_sample(
                alpha * th.ones(N)),
            requires_grad=True)
        self.Q = nn.Parameter(
            data=th.normal(
                0., 1., (D, H, N)), requires_grad=True)

    def forward(self, s):
        # spectrum is [B,D]
        # unbatched single head: Q is [D,H], QQT is [D,D]

        M = (self.Q @ self.pi.view(1, -1, 1)).squeeze(-1)
        MTs = M.transpose(0, 1).unsqueeze(0) @ s.unsqueeze(-1)
        MMTs = (M.unsqueeze(0) @ MTs).squeeze(-1)
        return MMTs


def apply_postprocessing(spec, model_d):
    # this is not a model method due to DataParallel

    if model_d["cfm_postprocessing"]:
        pp_spec = cfm_postprocess(spec, model_d["output_normalization"])
        return pp_spec
    else:
        return spec


def cfm_postprocess(spec, normalization, k=30, thresh=0.8):
    # this is only applied to predicted spectrum for validation

    norm_spec = F.normalize(spec, dim=1, p=1)
    topk, argtopk = th.topk(norm_spec, k=k, dim=1)
    cs = th.cumsum(topk, dim=1)
    keep_mask = th.roll(cs < thresh, dims=1, shifts=1)
    keep_mask[:, 0] = True
    keep_idx = th.nonzero(keep_mask, as_tuple=True)
    spec_idx = keep_idx[0]
    peak_idx = argtopk[keep_idx[0], keep_idx[1]]
    new_spec = th.zeros_like(spec)
    new_spec[spec_idx, peak_idx] = spec[spec_idx, peak_idx]
    assert th.all((new_spec > 0.).sum(dim=1) <= k)
    # renormalize
    if normalization == "l1":
        new_spec = F.normalize(new_spec, dim=1, p=1)
    elif normalization == "l2":
        new_spec = F.normalize(new_spec, dim=1, p=2)
    return new_spec
