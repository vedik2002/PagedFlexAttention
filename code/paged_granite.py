import torch
from torch import nn
from typing import Optional, Tuple
from paged_attention import PagedAttention
from transformers import AutoModelForCausalLM
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers.models.granite.modeling_granite import apply_rotary_pos_emb

class PagedGraniteAttention(nn.Module):
    def __init__(self, base_attn, config, num_pages, page_size, max_batch_size, device='cuda'):
        super().__init__()
        self.base_attn = base_attn
        self.config = config
        self.paged_attn = PagedAttention(
            n_pages=num_pages, page_size=page_size,
            max_batch_size=max_batch_size, device=device
        )

        # Architecture Details
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # 1. CRITICAL: Scale is attention_multiplier (usually 1/head_dim, e.g., 0.015625)
        # Granite uses this INSTEAD of 1/sqrt(d).
        self.scaling = getattr(config, "attention_multiplier", self.head_dim**-1)
        self.logits_soft_cap = getattr(config, "attention_logit_softcapping", None)

        # Physical Caches
        self.k_cache = torch.zeros(
            1, self.num_kv_heads, num_pages * page_size, self.head_dim,
            dtype=base_attn.q_proj.weight.dtype, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    @staticmethod
    def build_segment_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Build segment IDs for packed batches.
        attention_mask: (B, S, K, V), where K=V=S for causal, 1 for valid tokens, 0 for padding.
        Returns segment_ids: (B, S) with segment indices.
        """
        B, _, S, _ = attention_mask.shape
        device = attention_mask.device

        segment_ids = torch.zeros((B, S), dtype=torch.long, device=device)
        for b in range(B):
            seg_id = 0
            for s in range(S):
                if attention_mask[b, 0, s, :].sum() == 0:
                    seg_id += 1
                segment_ids[b, s] = seg_id
        return segment_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, S, _ = hidden_states.shape
        device = hidden_states.device

        # Handle position metadata
        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).repeat(B, 1)
            else:
                position_ids = torch.arange(S, device=device).unsqueeze(0).repeat(B, 1)

        # if attention_mask is not None:
        #     segment_ids = self.build_segment_ids(attention_mask)
        #     position_ids = segment_ids * S + position_ids
        kv_lens = position_ids[:, -1] + 1

        # 2. Linear Projections (No manual scaling here!)
        q = self.base_attn.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.base_attn.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.base_attn.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE (On unscaled q and k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # apply_rotary_pos_emb handles broadcasting internally
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Update the Paged Cache
        batch_indices = kwargs.get("batch_idx", torch.arange(B, device=device))
        # batch_indices = kwargs.get("batch_idx", torch.arange(B, device=device))

        for i in range(B):
            self.paged_attn.reserve(int(batch_indices[i].item()), kv_lens[i])

        # for b_idx in batch_indices:
        #     self.paged_attn.reserve(b_idx, kv_lens[b_idx])
        self.paged_attn.assign(batch_indices, position_ids, k, v, self.k_cache, self.v_cache)

        # 5. Score Modification
        p_to_l = self.paged_attn.physical_to_logical
        pg_size = self.paged_attn.page_size
        NEG_INF = torch.finfo(q.dtype).min

        def paged_score_mod(score, b, h, q_idx, kv_idx):
            target_batch = batch_indices[b]
            logical_block = p_to_l[target_batch, kv_idx // pg_size]
            logical_kv_idx = (logical_block * pg_size) + (kv_idx % pg_size)

            # Mask: Unallocated page OR padding/future token OR causal boundary
            is_valid = (logical_block >= 0) & (logical_kv_idx < kv_lens[b]) & (q_idx >= logical_kv_idx)

            # Apply Softcapping if defined in config
            if self.logits_soft_cap is not None:
                score = score / self.logits_soft_cap
                score = torch.tanh(score)
                score = score * self.logits_soft_cap

            return torch.where(is_valid, score, NEG_INF)

        # 6. Execute Flex Attention with correct scaling
        # attn_output = flex_attention(
        #     q, self.k_cache, self.v_cache,
        #     score_mod=paged_score_mod,
        #     scale=self.scaling, # Use the attention_multiplier as the scale
        #     enable_gqa=(self.num_heads != self.num_kv_heads)
        # )
        attn_output = self.paged_attn.forward(
            query=q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            kv_lens=kv_lens,
            batch_indices=batch_indices,
            score_mod=None,  # causal handled internally
            scale=self.scaling,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )


        # 7. Output Projection (Return exactly 2 values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        output = self.base_attn.o_proj(attn_output)

        return output, None
    

class PagedGranite(AutoModelForCausalLM):
    """
    GraniteForCausalLM with paged attention patched in-place.
    Acts as a first-class HF model.
    """
    def __init__(self, config):
        # HF constructs the model here (weights loaded later)
        super().__init__(config)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *,
            num_pages: int = 128,
            page_size: int = 64,
            max_batch_size: int = 1,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs,
        ):

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        cls._patch_granite_safely(
            model,
            num_pages=num_pages,
            page_size=page_size,
            max_batch_size=max_batch_size
        )

        return model

    @staticmethod
    def _patch_granite_safely(
            model,
            num_pages: int,
            page_size: int,
            max_batch_size: int
        ):
        count = 0
        for _, layer in enumerate(model.model.layers):
            # Check if it has a self_attn attribute and it's an attention class
            if hasattr(layer, "self_attn") and "Attention" in type(layer.self_attn).__name__:
                layer.self_attn = PagedGraniteAttention(
                    base_attn=layer.self_attn,
                    config=model.config,
                    num_pages=num_pages,
                    page_size=page_size,
                    max_batch_size=max_batch_size,
                    device="cuda"
                )
                count += 1
        print(f"Successfully patched {count} attention layers. Skipped Mamba layers.")

    @property
    def layers(self):
        """
        Returns the list of transformer layers.
        Works for both vanilla Granite and MoE Granite.
        """
        # AutoModelForCausalLM -> HF internal model
        internal_model = getattr(self, "model", None)
        if internal_model is None:
            raise AttributeError("Cannot find the internal model")

        if hasattr(internal_model, "layers"):
            # Vanilla Granite
            return internal_model.layers
        elif hasattr(internal_model, "model") and hasattr(internal_model.model, "layers"):
            # MoE Granite
            return internal_model.model.layers
        else:
            raise AttributeError("Cannot find transformer layers in model")

    @staticmethod
    def reset_cache(model):
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "paged_attn"):
                pa = attn.paged_attn

                pa.empty_pages = list(range(pa.n_pages - 1, -1, -1))
                pa.capacity.zero_()
                pa.page_table.fill_(-1)
                pa.physical_to_logical.fill_(-1)

                attn.k_cache.zero_()
                attn.v_cache.zero_()