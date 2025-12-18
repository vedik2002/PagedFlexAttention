import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from paged_granite import PagedGranite
from paged_attention import PagedAttention
from transformers import AutoModelForCausalLM

def test_paged_attention_correctness():
    print("=== Starting Paged Attention Test ===")

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available (FlexAttention requires CUDA).")
        return

    device = "cuda"
    torch.set_default_device(device)

    # --- 1. Configuration ---
    BATCH_SIZE = 4
    NUM_HEADS = 6
    HEAD_DIM = 64
    PAGE_SIZE = 16
    N_PAGES = 100  # Total physical pages available
    Q_LEN = 12     # Query length (e.g., chunked prefill or decoding steps)

    # Initialize Paged Attention Manager
    pa = PagedAttention(
        n_pages=N_PAGES,
        page_size=PAGE_SIZE,
        max_batch_size=BATCH_SIZE,
        device=device
    )

    # Allocate Physical Cache (Pre-allocated memory on GPU)
    # Shape: [1, H, Total_Pages * Page_Size, D]
    # Flex attention expects the cache to be treated as a single flattened sequence
    physical_k_cache = torch.zeros(1, NUM_HEADS, N_PAGES * PAGE_SIZE, HEAD_DIM, device=device)
    physical_v_cache = torch.zeros(1, NUM_HEADS, N_PAGES * PAGE_SIZE, HEAD_DIM, device=device)

    # --- 2. Generate Random Data (Ragged Batch) ---
    # We will create random KV sequences of different lengths for each batch item.
    # We will verify correctness by comparing against standard PyTorch attention.

    # Random lengths between 30 and 150 (spanning multiple pages)
    # Ensure they are > Q_LEN for this test to be interesting
    kv_lengths = torch.randint(Q_LEN + 5, 80, (BATCH_SIZE,), device=device)

    print(f"Test Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  KV Lengths: {kv_lengths.tolist()}")
    print(f"  Query Len:  {Q_LEN}")
    print(f"  Page Size:  {PAGE_SIZE}")

    queries = torch.randn(BATCH_SIZE, NUM_HEADS, Q_LEN, HEAD_DIM, device=device)

    # Store ground truth K and V to run reference implementation later
    ground_truth_ks = []
    ground_truth_vs = []

    # --- 3. Populate Paged Attention ---
    print("\nPopulating Page Table and Cache...")

    for i in range(BATCH_SIZE):
        seq_len = kv_lengths[i].item()

        # Generate random K/V for this sequence
        k_data = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM, device=device)
        v_data = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM, device=device)

        ground_truth_ks.append(k_data.squeeze(0))
        ground_truth_vs.append(v_data.squeeze(0))

        # 1. Reserve pages
        batch_idx_tensor = torch.tensor([i], device=device)
        seq_len_tensor = torch.tensor([seq_len], device=device)
        pa.reserve(batch_idx_tensor, seq_len_tensor)

        # 2. Assign data
        # We assign the whole sequence at once for this test.
        # assign expects input_pos of shape [B, S]
        input_pos = torch.arange(0, seq_len, device=device).unsqueeze(0) # [1, S]

        pa.assign(
            batch_idx=batch_idx_tensor,
            input_pos=input_pos,
            k_val=k_data, # [1, H, S, D]
            v_val=v_data, # [1, H, S, D]
            k_cache=physical_k_cache,
            v_cache=physical_v_cache
        )

    # --- 4. Run Paged Attention Forward Pass ---
    print("Running Paged Attention Forward...")

    # Flex Attention scaling default is 1/sqrt(D)
    scale = 1.0 / (HEAD_DIM ** 0.5)

    # We use a standard Causal Mask implicitly handled inside the PagedAttention.forward
    # when block_mask is None.
    # Note: In a real "prefill+decoding" scenario, Q usually aligns with the END of K.
    # Here, we are simulating that Q is attending to the *entire* K history provided.

    paged_output = pa.forward(
        query=queries,
        k_cache=physical_k_cache,
        v_cache=physical_v_cache,
        kv_lens=kv_lengths,
        scale=scale
    )

    # --- 5. Run Reference Implementation (SDPA) ---
    print("Running Reference (Standard SDPA)...")

    max_diff = 0.0

    for i in range(BATCH_SIZE):
        q_i = queries[i]      # [H, Q, D]
        k_i = ground_truth_ks[i] # [H, S, D]
        v_i = ground_truth_vs[i] # [H, S, D]

        # Standard Scaled Dot Product Attention
        # We need a causal mask.
        # In this specific test setup:
        # Query (Length Q) attends to Key (Length S).
        # Since we generated S > Q, and usually Q represents the *newest* tokens,
        # we need to define the causal relationship.
        #
        # For simplicity in this test, we will assume standard broadcasting causal mask
        # is NOT applied rigidly because S != Q. We simply want to attend to all available keys
        # up to the length defined.
        # HOWEVER, PagedAttention default mask in previous code was: `q_idx >= k_idx`.
        # This implies standard causal masking.

        # Let's align reference with the Logic in PagedAttention:
        # q_idx (0..Q) attends to k_idx (0..S). Mask is True if q_idx >= k_idx.

        # Create explicit mask for SDPA
        # Shape: [Q, S]
        q_idx = torch.arange(Q_LEN, device=device).unsqueeze(1)
        k_idx = torch.arange(kv_lengths[i].item(), device=device).unsqueeze(0)
        attn_mask = (q_idx >= k_idx) # Boolean mask

        # SDPA expects float mask for add (0, -inf) or boolean (True=Keep, False=Drop)
        # But torch.nn.functional.scaled_dot_product_attention handles boolean is_causal
        # ONLY if S == Q. Since S != Q, we pass explicit mask.

        ref_out = F.scaled_dot_product_attention(
            q_i.unsqueeze(0),
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
            scale=scale
        )

        ref_out = ref_out.squeeze(0) # [H, Q, D]

        # Compare
        pa_out_i = paged_output[i]

        diff = (ref_out - pa_out_i).abs().max().item()
        max_diff = max(max_diff, diff)

        if diff > 1e-3:
            print(f"Batch {i} FAILED. Max Diff: {diff}")
        else:
            print(f"Batch {i} PASSED. Max Diff: {diff:.6f}")

    print(f"\nTest Complete. Maximum discrepancy across batch: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("SUCCESS: Paged Attention matches Reference Implementation.")
    else:
        print("FAILURE: Differences detected.")

def detailed_parity_test(model_original, model_patched, text="Hello, my name is IBM Granite."):
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-micro")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    print("=" * 80)
    print("DIAGNOSTIC TEST START")
    print("=" * 80)

    # ------------------------------------------------------------------
    # TEST 0: Paged vs normal KV cache check  ✅ NEW
    # ------------------------------------------------------------------
    for i, layer in enumerate(model_patched.model.layers):
        attn = layer.self_attn

        # 1️⃣ Ensure normal HF KV cache is not present
        assert not hasattr(attn, "past_key_value"), f"❌ Layer {i}: normal past_key_value exists"

        # 2️⃣ Ensure paged attention cache exists
        assert hasattr(attn, "paged_attn"), f"❌ Layer {i}: paged_attn missing"
        assert hasattr(attn, "k_cache") and hasattr(attn, "v_cache"), f"❌ Layer {i}: paged K/V cache missing"

    print("✅ Paged attention cache confirmed. Normal KV cache disabled.")

    # ------------------------------------------------------------------
    # TEST 1: Model modes
    # ------------------------------------------------------------------
    print("\nTEST 1: Model Modes")
    print(f"Original training mode: {model_original.training}")
    print(f"Patched training mode: {model_patched.training}")

    # ------------------------------------------------------------------
    # TEST 2: Embedding parity
    # ------------------------------------------------------------------
    with torch.no_grad():
        hidden_orig = model_original.model.embed_tokens(inputs["input_ids"])
        hidden_patched = model_patched.model.embed_tokens(inputs["input_ids"])

    print("\nTEST 2: Embedding Parity")
    print(f"Max embedding diff: {(hidden_orig - hidden_patched).abs().max().item():.6e}")

    # ------------------------------------------------------------------
    # TEST 3: QKV weight parity
    # ------------------------------------------------------------------
    layer_idx = 0
    orig_attn = model_original.model.layers[layer_idx].self_attn
    patched_attn = model_patched.model.layers[layer_idx].self_attn

    print("\nTEST 3: Projection Weight Parity")
    print(f"Q diff: {(orig_attn.q_proj.weight - patched_attn.base_attn.q_proj.weight).abs().max().item():.6e}")
    print(f"K diff: {(orig_attn.k_proj.weight - patched_attn.base_attn.k_proj.weight).abs().max().item():.6e}")
    print(f"V diff: {(orig_attn.v_proj.weight - patched_attn.base_attn.v_proj.weight).abs().max().item():.6e}")
    print(f"O diff: {(orig_attn.o_proj.weight - patched_attn.base_attn.o_proj.weight).abs().max().item():.6e}")

    # ------------------------------------------------------------------
    # TEST 4: QKV activation parity
    # ------------------------------------------------------------------
    with torch.no_grad():
        q_o = orig_attn.q_proj(hidden_orig)
        k_o = orig_attn.k_proj(hidden_orig)
        v_o = orig_attn.v_proj(hidden_orig)

        q_p = patched_attn.base_attn.q_proj(hidden_patched)
        k_p = patched_attn.base_attn.k_proj(hidden_patched)
        v_p = patched_attn.base_attn.v_proj(hidden_patched)

    print("\nTEST 4: QKV Activation Parity")
    print(f"Q diff: {(q_o - q_p).abs().max().item():.6e}")
    print(f"K diff: {(k_o - k_p).abs().max().item():.6e}")
    print(f"V diff: {(v_o - v_p).abs().max().item():.6e}")

    # ------------------------------------------------------------------
    # RESET CACHE
    # ------------------------------------------------------------------
    for layer in model_patched.model.layers:
        attn = layer.self_attn
        attn.k_cache.zero_()
        attn.v_cache.zero_()
        attn.paged_attn.page_table.fill_(-1)
        attn.paged_attn.capacity.zero_()
        attn.paged_attn.physical_to_logical.fill_(-1)
        attn.paged_attn.empty_pages = list(range(attn.paged_attn.n_pages - 1, -1, -1))

    # ------------------------------------------------------------------
    # TEST 5: Full forward pass
    # ------------------------------------------------------------------
    print("\nTEST 5: Full Forward Pass")

    with torch.no_grad():
        out_orig = model_original(**inputs).logits
        out_patched = model_patched(**inputs).logits

    # ------------------------------------------------------------------
    # TEST 6: POST-FORWARD cache validation
    # ------------------------------------------------------------------
    print("\nTEST 6: Post-Forward Cache State")

    attn0 = model_patched.model.layers[0].self_attn
    print(f"K cache nonzero: {(attn0.k_cache != 0).sum().item()}")
    print(f"V cache nonzero: {(attn0.v_cache != 0).sum().item()}")
    print(f"Capacity: {attn0.paged_attn.capacity.item()}")
    print(f"Pages allocated: {(attn0.paged_attn.page_table >= 0).sum().item()}")

    # ------------------------------------------------------------------
    # TEST 7: Output difference
    # ------------------------------------------------------------------
    diff = (out_orig - out_patched).abs()

    print("\nTEST 7: Output Difference")
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print("\nDIAGNOSIS")

    max_diff = diff.max().item()

    if max_diff < 0.2:
        print("✅ PASS: fp16-consistent output (expected drift)")
    else:
        print("❌ FAIL: Unexpected numerical divergence")

    return max_diff

if __name__ == "__main__":
    test_paged_attention_correctness()
    model = PagedGranite.from_pretrained(
        "ibm-granite/granite-4.0-micro",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model_orig = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-4.0-micro",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    PagedGranite.reset_cache(model)
    detailed_parity_test(model_orig, model)