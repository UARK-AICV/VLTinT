import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-1):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        output: prediction_scores_list[idx].view(-1, self.config.vocab_size),
        target: input_labels_list[idx].view(-1)
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


class UnlikelihoodLoss(LabelSmoothingLoss):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-1, padding_idx=0):
        super().__init__(label_smoothing, tgt_vocab_size, ignore_index=ignore_index)
        self.padding_idx = padding_idx

    def forward(self, output, target):
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = F.kl_div(output, model_prob, reduction="sum")
        
        with torch.no_grad():
            # Make 'the triangle'.
            ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
            ctx_cands_ = (ctx_cands.tril(-1) + self.padding_idx)
            ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
            ctx_cands = ctx_cands.tril(-1) + ctx_cands_
            # Don't include the target for that timestep as a negative target.
            ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), self.padding_idx)
            negative_targets = torch.zeros_like(output).scatter_(1, ctx_cands, 1)
            negative_targets[:,self.padding_idx] = 0

        # Compute the unlikelihood loss
        one_minus_probs = torch.clamp((1.0 - output.exp()), min=1e-5)
        custom_loss = -torch.log(one_minus_probs)*negative_targets
        custom_loss = custom_loss.sum()
        norm = target.ne(self.padding_idx).sum().item()
        return loss.div(float(norm))+0.1*custom_loss.div(float(norm))


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_img = nn.Linear(768, 512, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, output, target):
        output = torch.mean(self.proj_img(output), dim=1)

        output = output / output.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * output @ target.t()

        labels = torch.arange(output.size(0), device=output.device)
        loss = F.cross_entropy(logits_per_image, labels)
        return loss


class SentLossDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_img = nn.Linear(768, 512, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, output, target, target_mask):
        output = output[target_mask]
        target = target[target_mask]

        output = torch.mean(self.proj_img(output), dim=1)

        output = output / output.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * output @ target.t()

        sent_prob = torch.nn.functional.softmax(target @ target.t(), dim=1)
        sent_prob[sent_prob < 0.01] = 0.0
        sent_prob[sent_prob > 0.99] = 1.0

        loss = F.kl_div(logits_per_image, sent_prob, reduction="sum")
        return loss


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class PositionEncodingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, n_filters=768, max_len=500):
        super().__init__()
        self.row_embed = nn.Embedding(max_len, n_filters)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)

    def forward(self, x):
        j = torch.arange(x.size(1), device=x.device)
        y_emb = self.row_embed(j)
        pos = y_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return x + pos


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len+max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len+max_v_len:] = \
        torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemory, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertLayerWithMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerWithMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        
        self.memory = HAM(config, config.max_v_len+config.max_t_len)
        self.memory_fuser = M2RF(config, config.max_v_len+config.max_t_len)
        self.memory_output = BertSelfOutput(config)

    def forward(self, prev_m, hidden_states, attention_mask=None):
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)
        
        if prev_m is None:
            bsz, tmprl_sz, ft_sz = hidden_states.shape
            prev_m = torch.zeros(bsz, tmprl_sz, 0, ft_sz, device=hidden_states.device)
        memory_fused = self.memory(hidden_states, prev_m)

        updated_m = torch.cat([prev_m, hidden_states.unsqueeze(2)], dim=2)

        fused_features = self.memory_fuser(hidden_states, memory_fused)

        layer_output = self.memory_output(fused_features, hidden_states)
        return updated_m, layer_output


class MARTLayerWithMemory(nn.Module):
    def __init__(self, config):
        super(MARTLayerWithMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.memory_initilizer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = BertSelfAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = BertOutput(config)

    def forward(self, prev_m, hidden_states, attention_mask):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)

        if prev_m is None:
            # only allow the initializer to see video part, not text part,
            # as it will be used for generation at current step
            init_memory_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initilizer(intermediate_output, init_memory_mask)  # (N, L, Di)

        # update memory, use raw attention_mask, no need to hide the text
        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)  # (N, M, Di)

        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # [(N, M, Di); (N, L, Di)] => [N, M+L, Di]
        bsz, n_memory_cells = prev_m.shape[:2]
        raw_memory_attention_mask = torch.cat(
            [attention_mask.new_ones(bsz, n_memory_cells), attention_mask], -1)  # (N, M+L)
        memory_attention_mask = make_pad_shifted_mask(
            raw_memory_attention_mask, max_v_len, max_t_len, memory_len=n_memory_cells)
        memory_attention_output = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, memory_attention_mask)  # (N, L, Di)
        memory_attention_output = self.memory_projection(memory_attention_output)  # (N, L, Di) -> (N, L, D)

        layer_output = self.output(memory_attention_output, attention_output)  # (N, L, D)

        return updated_m, layer_output


class BertEncoderWithMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderWithMemory, self).__init__()
        if hasattr(config, 'use_tint'):
            if config.use_tint:
                print("using TinT layer")
                self.layer = nn.ModuleList([BertLayerWithMemory(config) for _ in range(config.num_hidden_layers)])
            else:
                print("using MART layer")
                self.layer = nn.ModuleList([MARTLayerWithMemory(config) for _ in range(config.num_hidden_layers)])
        else:
            print("using TinT layer")
            self.layer = nn.ModuleList([BertLayerWithMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, prev_ms, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step. Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            prev_ms[layer_idx], hidden_states = layer_module(prev_ms[layer_idx], hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return prev_ms, all_encoder_layers


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    """A masked softmax module to correctly implement attention in Pytorch.
    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    
    Args:
        vector (torch.tensor): The tensor to softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.
        dim (int, optional): The dimension to softmax over.
                            Defaults to -1.
        memory_efficient (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax.
                                            Defaults to False.
        mask_fill_value ([type], optional): The value to fill masked values with if `memory_efficient` is `True`.
                                            Defaults to -1e32.
    
    Returns:
        torch.tensor: The masked softmaxed output
    """
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)
    return result


class Fuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, 
            dropout=config.attention_probs_dropout_prob
        )
        self.self_output = BertSelfOutput(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        

    def forward(self, src, src_mask=None, key_padding_mask=None):
        src2 = self.attention(
            src, src, src, attn_mask=src_mask,
            key_padding_mask=key_padding_mask
        )[0]
        if key_padding_mask is not None:
            src2 = src2.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        src = self.self_output(src2, src)
        src2 = self.intermediate(src)
        src = self.output(src2, src)
        if key_padding_mask is not None:
            src = src.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        return src


class HAM(nn.Module):
    def __init__(self, config, attention_steps=None):
        super().__init__()
        self.fuser = Fuser(config)
        self.attention_steps = config.max_v_len if attention_steps is None else attention_steps

        self.soft = True
        self.hard = False
        self.hybrid = False

    def forward(self, env_features=None, agent_features=None, agent_masks=None):
        if agent_features is None:
            return env_features

        if agent_masks is None:
            agent_masks = torch.ones(agent_features.shape[:3], device=agent_features.device)

        if env_features is None:
            input_mask_expanded = agent_masks.unsqueeze(-1).expand(agent_features.size()).float()
            return torch.sum(agent_features * input_mask_expanded, dim=2) / torch.clamp(input_mask_expanded.sum(2), min=1e-9)

        bsz, tmprl_sz, n_boxes, ft_sz = agent_features.shape

        # Fuse all agents together at every temporal point
        agent_fused_features = torch.zeros(bsz, tmprl_sz, ft_sz, device=agent_features.device)
        if n_boxes == 0:
            return agent_fused_features

        agent_env_feats = torch.unsqueeze(env_features, 2) + agent_features
        for smpl_bgn in range(0, tmprl_sz, self.attention_steps):
            smpl_end = smpl_bgn + self.attention_steps

            ae_feats = agent_env_feats[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes, ft_sz)  # bsz x n_boxes x feat_dim
            masks = agent_masks[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes)  # bsz x n_boxes

            if self.soft:
                hard_attn_masks = masks.bool()
            l2_norm = torch.norm(ae_feats, dim=-1)  # bsz x n_boxes
            l2_norm_softmax = masked_softmax(l2_norm, masks)  # bsz x n_boxes

            # Adaptive threshold is 1 / number of bounding boxes:
            ada_thresh = torch.clamp(1. / torch.sum(masks, dim=-1, keepdim=True), 0., 1.)

            # Generate hard attention masks
            if self.hard or self.hybrid:
                hard_attn_masks = l2_norm_softmax >= ada_thresh  # bsz x n_boxes

            keep_mask = (torch.sum(hard_attn_masks, dim=-1) > 0)  # bsz
            keep_indices = torch.masked_select(torch.arange(hard_attn_masks.size(0), device=agent_features.device), keep_mask)  # keep_mask

            fuser_input = agent_features[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes, ft_sz).permute(1, 0, 2)  # n_boxes x bsz x feat_dim

            if len(keep_indices) > 0:
                fuser_input = fuser_input[:, keep_indices]  # n_boxes x keep_mask x feat_dim
                hard_attn_masks = hard_attn_masks[keep_indices]  # keep_mask x n_boxes

                padded_output = torch.zeros(bsz * (smpl_end - smpl_bgn), ft_sz, device=agent_features.device)  # bsz x feat_dim
                if self.soft or self.hybrid:
                    fuser_output = self.fuser(fuser_input, key_padding_mask=~hard_attn_masks)  # n_boxes x keep_mask x feat_dim
                if self.hard:
                    fuser_output = fuser_input * hard_attn_masks.permute(1, 0).contiguous().unsqueeze(-1)
                fuser_output = torch.sum(fuser_output, dim=0) / torch.sum(hard_attn_masks, dim=-1, keepdim=True)  # keep_mask x feat_dim
                padded_output[keep_indices] = fuser_output
                agent_fused_features[:, smpl_bgn:smpl_end] = padded_output.view(bsz, -1, ft_sz)

        return agent_fused_features


class M2RF(nn.Module):
    def __init__(self, config, attention_steps=None):
        super().__init__()
        self.fuser = Fuser(config)
        self.attention_steps = config.max_v_len if attention_steps is None else attention_steps

    def forward(self, *features, device=None):
        to_cat_features = [i for i in features if i is not None]

        cat_features = torch.stack(to_cat_features, dim=2)
        bsz, tmprl_sz, ft_sz = to_cat_features[0].shape

        context_features = torch.zeros_like(to_cat_features[0], device=to_cat_features[0].device)
        for smpl_bgn in range(0, tmprl_sz, self.attention_steps):
            smpl_end = smpl_bgn + self.attention_steps

            fuser_input = cat_features[:, smpl_bgn:smpl_end].contiguous()
            fuser_input = fuser_input.view(-1, len(to_cat_features), ft_sz).permute(1, 0, 2)

            fuser_output = self.fuser(fuser_input)
            fuser_output = torch.mean(fuser_output, dim=0)
            context_features[:, smpl_bgn:smpl_end] = fuser_output.view(bsz, -1, ft_sz)
        return context_features


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.use_env = config.use_env
        self.use_agent = config.use_agent
        self.use_lang = config.use_lang
        if config.use_agent:
            self.ham_agent = HAM(config)
        if config.use_lang: 
            self.ham_lang = HAM(config)

        self.m2rf = M2RF(config)

        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        self.agent_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=config.max_position_embeddings)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape, f"{pretrained_embedding.shape} != {self.word_embeddings.weight.shape}"  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, input_ids, video_features, token_type_ids,
            agent_feature, agent_mask, lang_feature, lang_mask):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)
        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        env_embeddings, agent_embeddings, lang_embeddings = None, None, None
        if self.use_env:
            env_embeddings = self.video_embeddings(video_features)
        if self.use_agent:
            agent_embeddings = self.agent_embeddings(agent_feature)
            agent_embeddings = self.ham_agent(env_embeddings, agent_embeddings, agent_mask)
        if self.use_lang:
            lang_embeddings = self.word_fc(self.word_embeddings(lang_feature))
            lang_embeddings = self.ham_lang(env_embeddings, lang_embeddings, lang_mask)
        
        fused_embeddings = self.m2rf(env_embeddings, agent_embeddings, lang_embeddings)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings[:, :100] += fused_embeddings
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


class BertEmbeddingsTextUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsTextUntied, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_text = PositionEncoding(n_filters=config.hidden_size,
                                                         max_len=config.max_position_embeddings)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, text_input_ids):
        """ text_input_ids: (N, Lt) """
        words_embeddings = self.word_fc(self.word_embeddings(text_input_ids))  # (N, Lt, D)
        words_embeddings = self.position_embeddings_text(words_embeddings)
        return words_embeddings  # (N, Lt, D)


class BertEmbeddingsVideoUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsVideoUntied, self).__init__()
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_video = PositionEncoding(n_filters=config.hidden_size,
                                                          max_len=config.max_position_embeddings)

    def forward(self, video_features):
        """ video_features: (N, Lv, D) """
        video_embeddings = self.video_embeddings(video_features)  # (N, Lv, D)
        video_embeddings = self.position_embeddings_video(video_embeddings)
        return video_embeddings  # (N, Lv, D)


class BertLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, diagonal_mask=False):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        """
        self_attention_mask = attention_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = hidden_states.size(1)
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)
        attention_output = self.attention(hidden_states, self_attention_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemoryUntied(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, diagonal_mask=False, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
            output_all_encoded_layers:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertDecoderLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        """
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)  # (N, Lt, D)
        attention_output = self.norm1(attention_output + dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Is the attention mask correct?
        # Yes! Use the mask associated with key/value, not query. (query, key, value)
        # Additionally, there is no need to do subsequent masking, since each word has the right to see
        # all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            attention_output, enc_outputs, enc_outputs, enc_mask.unsqueeze(1))  # (N, Lt, D)
        dec_enc_attention_output = self.norm2(attention_output + dec_enc_attention_output)  # (N, Lt, D)

        # 3, linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return dec_enc_attention_output  # (N, Lt, D)


class BertDecoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertDecoderLayerNoMemoryUntied(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


class MemoryInitializer(nn.Module):
    def __init__(self, config):
        super(MemoryInitializer, self).__init__()
        # init memory
        self.n_memory_cells = config.n_memory_cells
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, config.n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            BertLayerNorm(config.hidden_size),
            nn.Dropout(config.memory_dropout_prob)
        )

    def forward(self, input_states, attention_mask):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            attention_mask: (N, L)
        """
        pooled_input_states = torch.sum(input_states * attention_mask.unsqueeze(-1), dim=1)   # (N, D)
        pooled_input_states = pooled_input_states / attention_mask.sum(1, keepdim=True)  # (N, D) no zero here
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory


class MemoryUpdater(nn.Module):
    def __init__(self, config):
        super(MemoryUpdater, self).__init__()
        self.memory_update_attention = BertSelfAttention(config)

        self.mc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.mz = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sz = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, prev_m, input_states, attention_mask):
        """ This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        """
        # memory attended inputs
        n_memory_cells = prev_m.shape[1]
        update_mask = attention_mask.unsqueeze(1).repeat(1, n_memory_cells, 1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)  # (N, M, D),

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))  # (N, M, D)

        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))  # (N, M, D)

        updated_memory = (1 - z_t) * c_t + z_t * prev_m  # (N, M, D)
        return updated_memory


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


class RecursiveTransformer(nn.Module):
    """ VLTinT model
    """
    def __init__(self, config):
        super(RecursiveTransformer, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddingsWithVideo(config, add_postion_embeddings=True)
        self.encoder = BertEncoderWithMemory(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1)
        self.clip_loss = SentLossDisc()
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(self, prev_ms, input_ids, video_features,
                    input_masks, token_type_ids, agent_feature,
                    agent_mask, lang_feature, lang_mask):
        """single step forward in the recursive structure"""
        embeddings = self.embeddings(
            input_ids, video_features, token_type_ids,
            agent_feature, agent_mask, lang_feature, lang_mask
        )  # (N, L, D)

        prev_ms, encoded_layer_outputs = self.encoder(
            prev_ms, embeddings, input_masks, output_all_encoded_layers=False)  # both outputs are list
        prediction_scores = self.decoder(encoded_layer_outputs[-1])  # (N, L, vocab_size)
        return prev_ms, encoded_layer_outputs, prediction_scores

    def forward(self, 
        input_ids_list,
        video_features_list,
        input_masks_list,
        token_type_ids_list,
        agent_feature_list,
        agent_mask_list,
        lang_feature_list,
        lang_mask_list,
        input_labels_list,
        sent_feat_list=None,
        return_memory=False):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits, `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions,
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,
        """
        # [(N, M, D)] * num_hidden_layers, initialized internally
        prev_ms = [None] * self.config.num_hidden_layers
        step_size = len(input_ids_list)
        memory_list = []  # [(N, M, D)] * num_hidden_layers * step_size
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        for idx in range(step_size):
            prev_ms, encoded_layer_outputs, prediction_scores = \
                    self.forward_step(prev_ms, input_ids_list[idx], video_features_list[idx],
                                    input_masks_list[idx], token_type_ids_list[idx], agent_feature_list[idx],
                                    agent_mask_list[idx], lang_feature_list[idx], lang_mask_list[idx])

            memory_list.append(prev_ms)
            encoded_outputs_list.append(encoded_layer_outputs)
            prediction_scores_list.append(prediction_scores)

        if return_memory:  # used to analyze memory
            return memory_list
        else:
            # compute loss, get predicted words
            caption_loss = 0.
            vis_feat_list = []
            sent_mask_list = []
            for idx in range(step_size):
                caption_loss += self.loss_func(prediction_scores_list[idx].view(-1, self.config.vocab_size),
                                               input_labels_list[idx].view(-1))
                if sent_feat_list is not None:
                    vis_feat_list.append(encoded_outputs_list[idx][-1][:,:self.config.max_v_len])
                    sent_mask_list.append(torch.all(input_labels_list[idx] == -1, dim=1))
            if sent_feat_list is not None:  
                batch_size = len(sent_feat_list[0])
                vis_feat = torch.stack(vis_feat_list).view(step_size*batch_size, self.config.max_v_len, -1)
                sent_feat = torch.stack(sent_feat_list).view(step_size*batch_size, -1)
                sent_mask = torch.stack(sent_mask_list).view(step_size*batch_size)
                caption_loss += self.clip_loss(vis_feat, sent_feat, sent_mask)
            return caption_loss, prediction_scores_list
