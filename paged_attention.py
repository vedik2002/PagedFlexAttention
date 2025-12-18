import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    noop_mask,
    flex_attention,
    create_block_mask,
)

__all__ = ["PagedAttention"]


# def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
#     return (x + multiple - 1) // multiple


class PagedAttention:
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is split into tensor blocks of fixed length and cached in a compact way.
    Thus we can avoid redundant memory consumption due to varying kv length and
    support a larger batch size.
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
    ) -> None:
        # number of pages
        self.n_pages = n_pages

        # number of tokens per page
        self.page_size = page_size

        # page table: [batch, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )

        # capacity: batch_idx -> allocated sequence length
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)

        # index of empty pages that is available for allocation
        self.empty_pages = list(range(n_pages - 1, -1, -1))

        # mapping from physical page index to logical page index
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )
    def reserve(self, batch_idx: int, seq_len: torch.Tensor) -> None:
        seq_len = int(seq_len.item())
        cur_capacity = int(self.capacity[batch_idx].item())  # 🔴 FIX

        if seq_len <= cur_capacity:
            return

        needed = seq_len - cur_capacity
        num_pages = (needed + self.page_size - 1) // self.page_size

        if len(self.empty_pages) < num_pages:
            raise RuntimeError(
                f"requested {num_pages} pages but only {len(self.empty_pages)} available"
            )

        start_page = cur_capacity // self.page_size
        end_page = start_page + num_pages

        allocated_pages = torch.tensor(
            self.empty_pages[-num_pages:],
            device=self.page_table.device,
            dtype=torch.int64,
        )
        self.empty_pages = self.empty_pages[:-num_pages]

        self.page_table[batch_idx, start_page:end_page] = allocated_pages
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page, end_page, device=self.page_table.device
        )

        self.capacity[batch_idx] += num_pages * self.page_size

    def erase(self, batch_idx: int) -> None:
        allocated = self.page_table[batch_idx] != -1
        pages = self.page_table[batch_idx][allocated]

        self.capacity[batch_idx] = 0
        self.empty_pages += pages.tolist()
        self.physical_to_logical[batch_idx].fill_(-1)
        self.page_table[batch_idx].fill_(-1)

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:

        if k_val.requires_grad:
            raise RuntimeError("k_val must not require grad")

        B, H, S, K_D = k_val.shape
        V_D = v_val.shape[-1]

        # --- logical → physical mapping ---
        logical_block = input_pos // self.page_size
        logical_offset = input_pos % self.page_size

        # 🔴 FIX: batch-wise gather
        page_table = self.page_table[batch_idx]              # [B, n_pages]
        physical_block = torch.gather(
            page_table, 1, logical_block.to(torch.int64)
        )

        # --- SAFETY CHECK ---
        if (physical_block < 0).any():
            raise RuntimeError("assign() called before reserve()")

        addr = (physical_block * self.page_size + logical_offset).reshape(-1)

        # flatten KV
        k_val = k_val.permute(1, 0, 2, 3).reshape(1, H, B * S, K_D)
        v_val = v_val.permute(1, 0, 2, 3).reshape(1, H, B * S, V_D)

        k_cache[:, :, addr, :] = k_val
        v_cache[:, :, addr, :] = v_val

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: torch.Tensor | None = None,
        kv_len: torch.Tensor | None = None,
    ) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
            kv_len (Optional[Tensor]): actual KV sequence length for upper bound check;
                shape :math:`(B,)` to handle multiple batches.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_size"
                f"but got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)
        page_table = self.page_table[batch_idx]

        new_kv_num_blocks = block_mask.kv_num_blocks.clone()

        # The physical page table might be larger than the max logical blocks
        # but the block mask only cares about MAX_BLOCKS_IN_COL.
        # We assume the logical mask is sparse enough.
        new_kv_indices = torch.zeros(
            (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
        )

        # We gather the physical page indices using the logical block indices
        # provided by the input block_mask.
        new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
            torch.gather(
                page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64)
            )
            .view(block_mask.kv_indices.shape)
            .to(torch.int32)
        )

        new_full_kv_indices, new_full_kv_num_blocks = None, None
        if block_mask.full_kv_num_blocks is not None:
            if block_mask.full_kv_indices is None:
                raise AssertionError(
                    "block_mask.full_kv_indices must not be None when full_kv_num_blocks is not None"
                )
            new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone()
            new_full_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(
                    page_table,
                    1,
                    block_mask.full_kv_indices.view(B, -1).to(torch.int64),
                )
                .view(block_mask.full_kv_indices.shape)
                .to(torch.int32)
            )

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, kv_len)

        # The K/V cache sent to flex_attention acts as one giant sequence of length n_pages * page_size.
        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)

        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_mask_mod(
        self,
        mask_mod: _mask_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(is_valid, mask_mod(b, h, q_idx, logical_kv_idx), False)

        return new_mask_mod

    def get_score_mod(
        self,
        score_mod: _score_mod_signature | None,
        kv_len: torch.Tensor | None = None,
        is_causal: bool = True, # Add this flag
    ) -> _score_mod_signature:
        if score_mod is None:
            score_mod = _identity

        def new_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            # 1. Map Physical to Logical
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset

            # 2. VALIDITY CHECKS
            # Block must be allocated to this batch
            live_block = logical_block_idx >= 0
            # Must be within the actual sequence length (ignore garbage in the rest of the page)
            within_bounds = (logical_kv_idx < kv_len[b]) if kv_len is not None else True
            # Causal check (MUST use logical indices)
            causal_ok = (q_idx >= logical_kv_idx) if is_causal else True

            is_valid = live_block & within_bounds & causal_ok

            return torch.where(
                is_valid,
                score_mod(score, b, h, q_idx, logical_kv_idx),
                float("-inf"),
            )

        return new_score_mod

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_lens: torch.Tensor,
        block_mask: BlockMask | None = None,
        batch_indices: torch.Tensor | None = None,
        score_mod: _score_mod_signature | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
    ):
        """
        Computes the paged attention.

        Args:
            query (Tensor): Query tensor; shape :math:`(B, H, Q_LEN, D)`.
            k_cache (Tensor): Physical key cache; shape :math:`(1, H, TOTAL_PAGES * PAGE_SIZE, D)`.
            v_cache (Tensor): Physical value cache; shape :math:`(1, H, TOTAL_PAGES * PAGE_SIZE, D)`.
            kv_lens (Tensor): The logical length of the KV sequence for each batch element.
                                Used to mask out padding and future tokens. Shape :math:`(B,)`.
            block_mask (BlockMask, optional): Logical block mask. If None, a standard causal mask is assumed
                                                for the logical sequence.
            batch_indices (Tensor, optional): The indices in the page table corresponding to the query batch.
                                                If None, assumes 0..B-1.
            score_mod (callable, optional): Logical score modification function.
            scale (float, optional): Attention scale factor.
            enable_gqa (bool): Enable Grouped Query Attention.
        """
        B, H, Q_LEN, D = query.shape
        device = query.device

        if batch_indices is None:
            batch_indices = torch.arange(B, device=device)

        # 1. Create a logical block mask if one isn't provided.
        # We assume a standard Causal mask over the maximum logical KV length.
        if block_mask is None:
            max_kv_len = kv_lens.max().item()

            def causal_mask(b, h, q, k):
                return q >= k

            # Create a block mask for the logical dimensions: (B, H, Q_LEN, Max_KV)
            # Note: We must ensure the BLOCK_SIZE matches the page_size.
            block_mask = create_block_mask(
                causal_mask,
                B=B,
                H=H,
                Q_LEN=Q_LEN,
                KV_LEN=max_kv_len,
                device=device,
                BLOCK_SIZE=self.page_size,
                _compile=True # Often improves performance for mask creation
            )

        # 2. Convert the logical block mask to a physical block mask.
        # This re-maps the indices in the block mask to point to the correct physical pages.
        physical_block_mask = self.convert_logical_block_mask(
            block_mask,
            batch_idx=batch_indices,
            kv_len=kv_lens
        )

        # 3. Handle Score Mods
        # Wrap the user score_mod (which expects logical indices) to handle physical indices.
        physical_score_mod = self.get_score_mod(score_mod, kv_lens)

        # 4. Call flex_attention
        # Note: k_cache and v_cache have batch size 1, but query has batch size B.
        # The block mask handles the routing (preventing batch B from seeing batch A's pages).
        output = torch.compile(flex_attention)(
            query,
            k_cache,
            v_cache,
            block_mask=physical_block_mask,
            score_mod=physical_score_mod,
            scale=scale,
            enable_gqa=enable_gqa
        )

        return output