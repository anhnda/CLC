import inspect
import math
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn

from flatquant.quant_utils import ActivationQuantizer
from flatquant.utils import skip_initialization
from flatquant.function_utils import get_init_scale, get_decompose_dim
from flatquant.trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix, InvDecomposeTransMatrix
from flatquant.flat_linear import FlatQuantizedLinear

from transformers.models.mistral.modeling_mistral import MistralMLP, MistralAttention, MistralDecoderLayer, MistralRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

try:
    from transformers.models.mistral.modeling_mistral import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None
    eager_attention_forward = None


class FlatQuantMistralMLP(MistralMLP):
    def __init__(self, args, module: MistralMLP):
        if hasattr(module, "config"):
            config = module.config
        else:
            config = SimpleNamespace(
                hidden_size=module.up_proj.in_features,
                intermediate_size=module.up_proj.out_features,
                hidden_act="silu",
            )
        super().__init__(config)
        self.args = args
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            up_device = self.up_proj.linear.weight.device
            down_device = self.down_proj.linear.weight.device
            self.up_smax = torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0], device=up_device) * 1e-5
            self.down_smax = torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0], device=down_device) * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.args.add_diag)
            down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.args.add_diag)
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = self.act_fn(gate_states) * up_states
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        if self.diag_init == "sq_style":
            if self.up_smax.device != x.device or self.up_smax.dtype != x.dtype:
                self.up_smax = self.up_smax.to(device=x.device, dtype=x.dtype)
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            if self.down_smax.device != x.device or self.down_smax.dtype != x.dtype:
                self.down_smax = self.down_smax.to(device=x.device, dtype=x.dtype)
            self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class FlatQuantMistralAttention(MistralAttention):
    @staticmethod
    def _build_rotary_embedding(module, head_dim):
        config = module.config
        weight_device = module.q_proj.weight.device
        attempts = [
            lambda: MistralRotaryEmbedding(config=config),
            lambda: MistralRotaryEmbedding(config),
            lambda: MistralRotaryEmbedding(
                head_dim,
                max_position_embeddings=getattr(config, "max_position_embeddings", 2048),
                base=getattr(config, "rope_theta", 10000),
                device=weight_device,
            ),
            lambda: MistralRotaryEmbedding(
                head_dim,
                getattr(config, "max_position_embeddings", 2048),
                getattr(config, "rope_theta", 10000),
                weight_device,
            ),
            lambda: MistralRotaryEmbedding(head_dim),
        ]
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except (TypeError, AttributeError, IndexError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unable to construct MistralRotaryEmbedding")
    def __init__(self, args, module: MistralAttention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        self._use_position_embeddings_api = "position_embeddings" in inspect.signature(module.forward).parameters
        self.num_heads = getattr(module, "num_heads", module.config.num_attention_heads)
        self.num_key_value_heads = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
        self.head_dim = getattr(module, "head_dim", getattr(module.config, "head_dim", module.config.hidden_size // module.config.num_attention_heads))
        self.num_key_value_groups = getattr(module, "num_key_value_groups", self.num_heads // self.num_key_value_heads)
        self.scaling = getattr(module, "scaling", self.head_dim ** -0.5)
        self.attention_dropout = getattr(module, "attention_dropout", getattr(module.config, "attention_dropout", 0.0))
        self.is_causal = getattr(module, "is_causal", True)
        if hasattr(module, "rotary_emb"):
            self.rotary_emb = module.rotary_emb
        elif self._use_position_embeddings_api:
            self.rotary_emb = self._build_rotary_embedding(module, self.head_dim)

        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, sym=not(args.q_asym), lac=args.lac, groupsize=-1)
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, sym=not(args.k_asym), lac=args.lac, groupsize=-1)
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, sym=not(args.v_asym), lac=args.lac, groupsize=-1)

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            q_device = self.q_proj.linear.weight.device
            self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0], device=q_device) * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=self.args.add_diag)
            self.o_trans = SingleTransMatrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SingleTransMatrix(head_dim)
        else:
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            if self.ln_smax.device != hidden_states.device or self.ln_smax.dtype != hidden_states.dtype:
                self.ln_smax = self.ln_smax.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self.ln_smax = torch.maximum(self.ln_smax, hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def _project_qkv(self, hidden_states):
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return input_shape, query_states, key_states, value_states

    def _project_output(self, attn_output):
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        if self._ori_mode:
            return self.o_proj._ori_forward(attn_output)
        if self.o_trans is None and self.vcache_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.head_dim)
            attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
            return self.o_proj(attn_output)

        init_shape = attn_output.shape
        attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.head_dim)
        attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
        if not self._eval_mode:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            return self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
        return self.o_proj(attn_output)

    def _manual_attention(self, query_states, key_states, value_states, attention_mask):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        expected_shape = (query_states.shape[0], self.num_heads, query_states.shape[2], self.head_dim)
        if attn_output.size() != expected_shape:
            raise ValueError(f"`attn_output` should be of size {expected_shape}, but is {attn_output.size()}")
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape, query_states, key_states, value_states = self._project_qkv(hidden_states)

        cache_obj = past_key_values if past_key_values is not None else past_key_value

        if self._use_position_embeddings_api:
            if position_embeddings is None:
                position_embeddings = kwargs.pop('position_embeddings', None)
            if position_embeddings is None:
                if position_ids is None:
                    raise ValueError('position_embeddings or position_ids must be provided for this Mistral attention implementation')
                cos, sin = self.rotary_emb(value_states, position_ids)
                position_embeddings = (cos, sin)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if cache_obj is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx, cache_kwargs)

        used_attention_interface = self._use_position_embeddings_api and eager_attention_forward is not None and ALL_ATTENTION_FUNCTIONS is not None
        if used_attention_interface:
            attention_interface: Callable = eager_attention_forward
            if getattr(self.config, '_attn_implementation', 'eager') != 'eager':
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=getattr(self.config, 'sliding_window', None),
                **kwargs,
            )
        else:
            attn_output, attn_weights = self._manual_attention(query_states, key_states, value_states, attention_mask)

        if not used_attention_interface:
            attn_output = attn_output.transpose(1, 2)
        attn_output = self._project_output(attn_output)

        if not output_attentions:
            attn_weights = None

        if self._use_position_embeddings_api:
            return attn_output, attn_weights
        return attn_output, attn_weights, cache_obj

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, 'ln_smax')
        qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


class FlatQuantMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, args, module: MistralDecoderLayer):
        super().__init__(module.self_attn.config, module.self_attn.layer_idx)
        self.hidden_size = getattr(module, "hidden_size", module.self_attn.config.hidden_size)
        self.self_attn = FlatQuantMistralAttention(args, module.self_attn)
        self.mlp = FlatQuantMistralMLP(args, module.mlp)
        self.input_layernorm = module.input_layernorm
        self.post_attention_layernorm = module.post_attention_layernorm

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        output_attentions=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
            attn_weights = attn_output[1] if len(attn_output) > 1 else None
            present_key_value = attn_output[2] if len(attn_output) > 2 else None
        else:
            hidden_states = attn_output
            attn_weights = None
            present_key_value = None
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def apply_flatquant_to_mistral(args, model):
    skip_initialization()
    for layer in range(model.config.num_hidden_layers):
        model.model.layers[layer] = FlatQuantMistralDecoderLayer(args, model.model.layers[layer])
    return model
