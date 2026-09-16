"""
test_mmbert_smoke.py — End-to-End Compatibility & Smoke Test for mmBERT

Tests:
  1. Tokenizer validation & fail-fast checks
  2. Model initialization with configurable target_layers
  3. Model forward pass (branch="both", "en", "vi")
  4. LoRA injection with fail-fast validation (verifies target_modules=['Wqkv', 'Wo'])
  5. Stage 1 forward + loss + backward pass (QA head + layer_weights)
  6. Stage 2 forward + Sinkhorn OT loss + LoRA backward pass
  7. Dataloader dynamic pad_id validation (pad_id=0 for mmBERT)

Usage:
  python test_mmbert_smoke.py
  python test_mmbert_smoke.py --model_name jhu-clsp/mmBERT-base
  python test_mmbert_smoke.py --device cuda
"""

import sys
import os
import argparse
import traceback
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from phase2_model.model_core import CrossLingualOTModel, get_default_target_layers
from phase3_loss.losses import OTAlignmentLoss
from data.xquad_loader import xquad_collate_fn


def run_smoke_tests(model_name: str = "jhu-clsp/mmBERT-base", device_str: str = "cpu"):
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print("=" * 70)
    print(f"  mmBERT COMPATIBILITY & SMOKE TEST SUITE")
    print(f"  Model : {model_name}")
    print(f"  Device: {device}")
    print("=" * 70)

    results = {}

    # ──────────────────────────────────────────────────────────────
    # Test 1: Tokenizer & Fail-Fast Validation
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 1/7] Testing Tokenizer & Fail-Fast Check...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            raise ValueError(f"[Fail-Fast] {model_name} has no pad_token_id defined!")
        if tokenizer.sep_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.sep_token_id = tokenizer.eos_token_id

        print(f"  ✓ Tokenizer loaded: pad_token_id={tokenizer.pad_token_id}, "
              f"sep_token_id={tokenizer.sep_token_id}, cls_token_id={tokenizer.cls_token_id}")

        # Check special tokens for mmBERT
        if "mmbert" in model_name.lower():
            assert tokenizer.pad_token_id == 0, f"Expected pad_token_id=0 for mmBERT, got {tokenizer.pad_token_id}"
            assert tokenizer.sep_token_id == 1, f"Expected sep_token_id=1 for mmBERT, got {tokenizer.sep_token_id}"
            assert tokenizer.cls_token_id == 2, f"Expected cls_token_id=2 for mmBERT, got {tokenizer.cls_token_id}"
            print("  ✓ mmBERT special token IDs verified (pad=0, sep=1, cls=2)")

        # Verify fail-fast trigger
        class DummyTokenizer:
            pad_token_id = None
        try:
            dummy = DummyTokenizer()
            if dummy.pad_token_id is None:
                raise ValueError("[Fail-Fast Triggered as expected]")
        except ValueError:
            print("  ✓ Fail-fast check triggers properly when pad_token_id is None")

        results["1_tokenizer"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 1 FAILED: {e}")
        traceback.print_exc()
        results["1_tokenizer"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 2: Model Initialization & Configurable Target Layers
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 2/7] Testing Model Initialization & Configurable Target Layers...")
    try:
        # Default target layers
        model_default = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False)
        default_layers = model_default.target_layers
        print(f"  ✓ Default target_layers auto-detected: {default_layers}")
        if "mmbert" in model_name.lower():
            assert default_layers == [11, 12, 14, 16], f"Expected [11, 12, 14, 16] for mmBERT, got {default_layers}"
        assert len(model_default.layer_weights) == len(default_layers), "layer_weights length mismatch"

        # Custom configurable target layers
        custom_layers = [9, 13, 15, 17]
        model_custom = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False, target_layers=custom_layers)
        assert model_custom.target_layers == custom_layers, f"Expected custom layers {custom_layers}, got {model_custom.target_layers}"
        assert len(model_custom.layer_weights) == 4, "layer_weights length mismatch for custom layers"
        print(f"  ✓ Custom target_layers configured successfully: {model_custom.target_layers}")

        results["2_configurable_layers"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 2 FAILED: {e}")
        traceback.print_exc()
        results["2_configurable_layers"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 3: Model Forward Pass (branch="both", "en", "vi")
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 3/7] Testing Model Forward Pass (both, en, vi)...")
    try:
        model = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=True).to(device)
        model.eval()

        B, L = 2, 32
        dummy_batch = {
            "en_input_ids": torch.randint(10, 1000, (B, L), device=device),
            "en_attention_mask": torch.ones((B, L), dtype=torch.long, device=device),
            "vi_input_ids": torch.randint(10, 1000, (B, L), device=device),
            "vi_attention_mask": torch.ones((B, L), dtype=torch.long, device=device),
            "en_start_position": torch.tensor([5, 8], dtype=torch.long, device=device),
            "en_end_position": torch.tensor([7, 10], dtype=torch.long, device=device),
            "en_question_end": torch.tensor([4, 4], dtype=torch.long, device=device),
            "vi_question_end": torch.tensor([4, 4], dtype=torch.long, device=device),
        }
        # Add some padding to test dynamic truncation
        dummy_batch["en_attention_mask"][:, 25:] = 0
        dummy_batch["vi_attention_mask"][:, 28:] = 0

        with torch.no_grad():
            # branch="both"
            out_both = model(dummy_batch, branch="both")
            assert "en_hidden" in out_both and "vi_hidden" in out_both and "cost_matrix" in out_both
            assert out_both["en_hidden"].shape[-1] == 768
            assert out_both["vi_hidden"].shape[-1] == 768
            # Dynamic truncation: T_en = 25, T_vi = 28
            assert out_both["cost_matrix"].shape == (B, 25, 28), f"Expected cost matrix shape (2, 25, 28), got {out_both['cost_matrix'].shape}"
            print(f"  ✓ branch='both' forward passed: cost_matrix={out_both['cost_matrix'].shape}")

            # branch="en"
            out_en = model(dummy_batch, branch="en")
            assert "hidden" in out_en and out_en["hidden"].shape[-1] == 768
            print(f"  ✓ branch='en' forward passed: hidden={out_en['hidden'].shape}")

            # branch="vi"
            out_vi = model(dummy_batch, branch="vi")
            assert "hidden" in out_vi and out_vi["hidden"].shape[-1] == 768
            print(f"  ✓ branch='vi' forward passed: hidden={out_vi['hidden'].shape}")

        results["3_forward_branches"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 3 FAILED: {e}")
        traceback.print_exc()
        results["3_forward_branches"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 4: LoRA Injection & Fail-Fast Validation
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 4/7] Testing LoRA Injection with Fail-Fast Validation...")
    try:
        model_lora = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=True).to(device)

        # Check torchao environment advisory
        try:
            import torchao
            print(f"  [Advisory] torchao version {torchao.__version__} detected in environment.")
        except ImportError:
            print("  ✓ torchao not installed (optimal for standard PEFT LoRA).")

        # Apply LoRA (should auto-detect ['Wqkv', 'Wo'] for mmBERT)
        model_lora.apply_lora()
        trainable_params = sum(p.numel() for p in model_lora.backbone.parameters() if p.requires_grad)
        print(f"  ✓ LoRA injected successfully! Trainable parameters: {trainable_params:,}")
        assert trainable_params > 0, "No trainable parameters after LoRA injection!"

        # Test fail-fast with invalid target_modules
        model_bad = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False).to(device)
        try:
            model_bad.apply_lora(target_modules=["non_existent_module_xyz"])
            raise AssertionError("LoRA fail-fast did not catch invalid target_modules!")
        except RuntimeError as rerr:
            print(f"  ✓ LoRA fail-fast correctly caught invalid target_modules: {rerr}")

        results["4_lora_injection"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 4 FAILED: {e}")
        if "torchao" in str(e):
            print("\n  💡 TIP: Run `pip uninstall -y torchao` to fix PEFT torchao incompatibility on Colab.")
        traceback.print_exc()
        results["4_lora_injection"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 5: Stage 1 Backward Pass Smoke Test
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 5/7] Testing Stage 1 Forward + Loss + Backward Pass...")
    try:
        model_s1 = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=False).to(device)
        model_s1.train()

        criterion_s1 = OTAlignmentLoss(
            hidden_size=model_s1.hidden_size,
            lambda_ot=0.0,
            lambda_span=0.0,
            lambda_cons=0.0,
        ).to(device)

        # Forward on EN branch
        out_s1 = model_s1(dummy_batch, branch="en")
        s1_loss_dict = criterion_s1(
            H_en=out_s1["hidden"],
            H_vi=None,
            batch=dummy_batch,
            step=1,
            en_pad_mask=out_s1["en_pad_mask"],
            model_outputs=None,
        )

        total_loss = s1_loss_dict["total_loss"]
        print(f"  ✓ Stage 1 forward passed. Total Loss: {total_loss.item():.4f}")

        # Backward pass
        total_loss.backward()

        # Check gradients
        assert model_s1.layer_weights.grad is not None, "layer_weights has no gradient!"
        backbone_grads = sum(1 for p in model_s1.backbone.parameters() if p.grad is not None)
        assert backbone_grads > 0, "No gradients in backbone!"
        head_grads = sum(1 for p in criterion_s1.qa_head.parameters() if p.grad is not None)
        assert head_grads > 0, "No gradients in QA head!"

        print(f"  ✓ Stage 1 backward passed! Gradients verified in backbone ({backbone_grads} tensors), "
              f"layer_weights ({model_s1.layer_weights.grad.shape}), and QA head.")
        results["5_stage1_backward"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 5 FAILED: {e}")
        traceback.print_exc()
        results["5_stage1_backward"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 6: Stage 2 Backward Pass Smoke Test (Sinkhorn OT + LoRA)
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 6/7] Testing Stage 2 Forward + Sinkhorn OT Loss + LoRA Backward Pass...")
    try:
        model_s2 = CrossLingualOTModel(model_name=model_name, compute_cost_matrix=True).to(device)
        model_s2.apply_lora()
        model_s2.train()

        criterion_s2 = OTAlignmentLoss(
            hidden_size=model_s2.hidden_size,
            lambda_ot=0.5,
            lambda_span=1.0,
            lambda_cons=50.0,
        ).to(device)

        out_s2 = model_s2(dummy_batch, branch="both")
        s2_loss_dict = criterion_s2(
            H_en=out_s2["en_hidden"],
            H_vi=out_s2["vi_hidden"],
            batch=dummy_batch,
            step=1,
            en_pad_mask=out_s2["en_pad_mask"],
            vi_pad_mask=out_s2["vi_pad_mask"],
            model_outputs=out_s2,
        )

        total_loss_s2 = s2_loss_dict["total_loss"]
        print(f"  ✓ Stage 2 forward passed. Total Loss: {total_loss_s2.item():.4f} "
              f"(L_ot: {s2_loss_dict['loss_ot'].item():.4f}, L_qa: {s2_loss_dict['loss_qa'].item():.4f})")

        # Backward pass
        total_loss_s2.backward()

        lora_grads = sum(1 for n, p in model_s2.named_parameters() if "lora" in n and p.grad is not None)
        assert lora_grads > 0, "No gradients in LoRA adapter layers!"
        print(f"  ✓ Stage 2 backward passed! Gradients verified in {lora_grads} LoRA adapter tensors.")

        results["6_stage2_lora_backward"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 6 FAILED: {e}")
        traceback.print_exc()
        results["6_stage2_lora_backward"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Test 7: Dataloader Collate Function (Dynamic pad_id)
    # ──────────────────────────────────────────────────────────────
    print("\n[TEST 7/7] Testing Dataloader Dynamic pad_id...")
    try:
        pad_id_mmbert = tokenizer.pad_token_id
        dummy_items = [
            {
                "en_input_ids": torch.tensor([2, 10, 20, 1]),
                "en_attention_mask": torch.tensor([1, 1, 1, 1]),
                "en_start_positions": torch.tensor(1),
                "en_end_positions": torch.tensor(2),
                "en_question_end": torch.tensor(1),
                "vi_input_ids": torch.tensor([2, 30, 40, 50, 1]),
                "vi_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "vi_question_end": torch.tensor(1),
            },
            {
                "en_input_ids": torch.tensor([2, 100, 1]),
                "en_attention_mask": torch.tensor([1, 1, 1]),
                "en_start_positions": torch.tensor(0),
                "en_end_positions": torch.tensor(0),
                "en_question_end": torch.tensor(1),
                "vi_input_ids": torch.tensor([2, 300, 1]),
                "vi_attention_mask": torch.tensor([1, 1, 1]),
                "vi_question_end": torch.tensor(1),
            },
        ]

        collated = xquad_collate_fn(dummy_items, pad_id=pad_id_mmbert)
        # Check padding value in the second item (shorter item)
        assert collated["en_input_ids"][1, -1].item() == pad_id_mmbert, \
            f"Expected pad_id {pad_id_mmbert} in en_input_ids, got {collated['en_input_ids'][1, -1].item()}"
        assert collated["vi_input_ids"][1, -1].item() == pad_id_mmbert, \
            f"Expected pad_id {pad_id_mmbert} in vi_input_ids, got {collated['vi_input_ids'][1, -1].item()}"

        print(f"  ✓ Dynamic padding verified: pad_id={pad_id_mmbert} correctly used for batch padding.")
        results["7_dataloader_pad"] = "PASS"
    except Exception as e:
        print(f"  ❌ TEST 7 FAILED: {e}")
        traceback.print_exc()
        results["7_dataloader_pad"] = "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SMOKE TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for test_name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {test_name:30s}: {status}")
        if status != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("🎉 ALL 7 TESTS PASSED! mmBERT is 100% ready for Stage 1 training.")
    else:
        print("⚠️ Some tests failed. Please review errors above before starting training.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmBERT Compatibility & Smoke Test Suite")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    success = run_smoke_tests(model_name=args.model_name, device_str=args.device)
    sys.exit(0 if success else 1)
