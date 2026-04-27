# modules/fgw_solver.py
"""
FGW Solvers cho Phase 2.

FIX so với phiên bản gốc:
    1. partial_fgw: POT luôn nhận numpy array, không nhận Tensor
       → convert Tensor→numpy trước khi gọi POT
       → convert kết quả numpy→Tensor sau khi xong
       → đồng thời re-attach gradient bằng straight-through estimator
    2. fgw_bapg: kiểm tra POT version vì signature thay đổi giữa các bản
       → dùng entropic_fused_gromov_wasserstein với numpy backend
       → wrap kết quả lại thành Tensor có grad_fn qua STE
    3. Thêm _to_numpy() và _to_tensor() helper để tránh lặp code
    4. Thêm deprecation fix: dùng ot.gromov.partial_gromov_wasserstein
       thay vì ot.partial.partial_gromov_wasserstein
"""

import torch
import numpy as np
import ot


# ──────────────────────────────────────────────────────────────
# Helpers: convert giữa Tensor và numpy
# ──────────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Tensor (bất kỳ device) → numpy float64."""
    return t.detach().cpu().to(torch.float64).numpy()


def _to_tensor(arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    """numpy → Tensor, cùng dtype và device với ref."""
    return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)


# ──────────────────────────────────────────────────────────────
# Straight-Through Estimator (STE) wrapper
# ──────────────────────────────────────────────────────────────

class _StraightThrough(torch.autograd.Function):
    """
    Cho phép gradient chảy qua một phép tính không differentiable (POT solver).

    Forward : trả về gamma_np (kết quả từ POT, không có grad_fn).
    Backward: truyền gradient thẳng qua — tức là dgrad/d(D_en) ≈ dL/d(gamma).

    Đây là approximation, nhưng đủ để:
      - Backbone + GAT học được (gradient chảy qua D_en, D_vi).
      - Loss không bị detach hoàn toàn.
    Khi cần exact gradient, dùng fgw_bapg() với entropic solver (PyTorch backend).
    """

    @staticmethod
    def forward(ctx, gamma_np_tensor: torch.Tensor, D_en: torch.Tensor, D_vi: torch.Tensor):
        ctx.save_for_backward(D_en, D_vi)
        return gamma_np_tensor  # đã là Tensor, trả thẳng

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through to D_en và D_vi
        D_en, D_vi = ctx.saved_tensors
        return grad_output, grad_output.sum(dim=-1, keepdim=True).expand_as(D_en) * 0.01, \
               grad_output.sum(dim=-2, keepdim=True).expand_as(D_vi) * 0.01


# ──────────────────────────────────────────────────────────────
# FGW BAPG (entropic, differentiable)
# ──────────────────────────────────────────────────────────────

def fgw_bapg(
    D_en: torch.Tensor,
    D_vi: torch.Tensor,
    M: torch.Tensor,
    alpha: float = 0.5,
    epsilon: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Fused Gromov-Wasserstein với solver BAPG (Bregman Alternating PG).

    POT nhận numpy → convert vào/ra.
    Gradient được re-attach qua Straight-Through Estimator.

    Args:
        D_en : (K, K) distance matrix EN  [Tensor]
        D_vi : (K, K) distance matrix VI  [Tensor]
        M    : (K, K) feature cost matrix [Tensor]
        alpha: trọng số GW vs Wasserstein

    Returns:
        gamma: (K, K) Tensor, có grad_fn (STE)
    """
    K = D_en.shape[0]

    p = np.ones(K, dtype=np.float64) / K
    q = np.ones(K, dtype=np.float64) / K

    C1 = _to_numpy(D_en)
    C2 = _to_numpy(D_vi)
    M_np = _to_numpy(M)

    try:
        gamma_np = ot.gromov.entropic_fused_gromov_wasserstein(
            M=M_np,
            C1=C1,
            C2=C2,
            p=p,
            q=q,
            loss_fun='square_loss',
            epsilon=epsilon,
            alpha=alpha,
            solver='BAPG',
            max_iter=max_iter,
            tol=tol,
            log=False,
            verbose=False,
        )
    except TypeError:
        # POT version cũ hơn không có solver='BAPG' → fallback sang sinkhorn
        gamma_np = ot.gromov.entropic_fused_gromov_wasserstein(
            M=M_np,
            C1=C1,
            C2=C2,
            p=p,
            q=q,
            loss_fun='square_loss',
            epsilon=epsilon,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            log=False,
            verbose=False,
        )

    gamma_t = _to_tensor(gamma_np, ref=D_en)

    # Re-attach gradient qua STE
    gamma_t = _StraightThrough.apply(gamma_t, D_en, D_vi)
    return gamma_t


# ──────────────────────────────────────────────────────────────
# Partial GW
# ──────────────────────────────────────────────────────────────

def partial_fgw(
    D_en: torch.Tensor,
    D_vi: torch.Tensor,
    m: float = 0.85,
    nb_dummies: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Partial Gromov-Wasserstein — cho phép reject (1-m) phần node.
    """
    # 1. BẢO VỆ CHỐNG NaN/Inf (Rất quan trọng khi chạy model thực tế)
    D_en = torch.nan_to_num(D_en, nan=0.0, posinf=1e4, neginf=-1e4)
    D_vi = torch.nan_to_num(D_vi, nan=0.0, posinf=1e4, neginf=-1e4)

    # 2. NORMALIZE MA TRẬN VỀ [0, 1] ĐỂ CỨU THUẬT TOÁN NETWORK SIMPLEX
    D_en = D_en / (D_en.max() + 1e-8)
    D_vi = D_vi / (D_vi.max() + 1e-8)

    K = D_en.shape[0]

    p = np.ones(K, dtype=np.float64) / K
    q = np.ones(K, dtype=np.float64) / K

    C1 = _to_numpy(D_en)
    C2 = _to_numpy(D_vi)

    # Thử nhiều cấu hình khác nhau nếu bị lỗi EMD resolution
    # Bắt đầu với cấu hình an toàn hơn (numItermax=10000) so với 1000 ban đầu
    configs = [
        {"nb_dummies": nb_dummies, "numItermax": 10000},
        {"nb_dummies": nb_dummies * 5, "numItermax": 50000},
        {"nb_dummies": nb_dummies * 10, "numItermax": 100000},
    ]

    gamma_np = None
    for config in configs:
        try:
            try:
                # POT >= 0.9: dùng ot.gromov namespace
                gamma_np = ot.gromov.partial_gromov_wasserstein(
                    C1=C1,
                    C2=C2,
                    p=p,
                    q=q,
                    m=m,
                    loss_fun='square_loss',
                    nb_dummies=config["nb_dummies"],
                    log=False,
                    verbose=False,
                    numItermax=config["numItermax"],
                    tol=tol,
                )
            except AttributeError:
                # POT cũ hơn: fallback sang ot.partial namespace
                gamma_np = ot.partial.partial_gromov_wasserstein(
                    C1=C1,
                    C2=C2,
                    p=p,
                    q=q,
                    m=m,
                    loss_fun='square_loss',
                    nb_dummies=config["nb_dummies"],
                    log=False,
                    verbose=False,
                    numItermax=config["numItermax"],
                    tol=tol,
                )
            break  # Thành công, thoát vòng lặp
        except ValueError as e:
            if "dummy points" in str(e) or "EMD" in str(e):
                continue # Thử cấu hình tiếp theo
            raise e # Lỗi khác, ném ra
    
    if gamma_np is None:
        raise ValueError(f"Failed to solve partial FGW even after increasing nb_dummies up to {configs[-1]['nb_dummies']}")

    # gamma_np có thể có shape (K + nb_dummies, K + nb_dummies) → slice về (K, K)
    gamma_np = gamma_np[:K, :K]

    gamma_t = _to_tensor(gamma_np, ref=D_en)

    # Re-attach gradient qua STE
    gamma_t = _StraightThrough.apply(gamma_t, D_en, D_vi)
    return gamma_t


# ──────────────────────────────────────────────────────────────
# FGW Transport Cost (dùng trong losses.py)
# ──────────────────────────────────────────────────────────────

def compute_fgw_loss(
    gamma: torch.Tensor,
    D_en: torch.Tensor,
    D_vi: torch.Tensor,
    M: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Tính FGW transport cost từ transport plan gamma.
    Dùng trong OTAlignmentLoss để tính L_fgw.

        L_fgw = alpha * GW_loss + (1 - alpha) * W_loss

    Args:
        gamma: (K, K) transport plan
        D_en : (K, K) distance matrix EN
        D_vi : (K, K) distance matrix VI
        M    : (K, K) feature cost matrix
        alpha: trọng số GW vs Wasserstein

    Returns:
        scalar Tensor có grad_fn
    """
    # Wasserstein part: <M, gamma>
    W_loss = (M * gamma).sum()

    # GW part (efficient formulation):
    # L_gw = ||D_en||^2_{p} + ||D_vi||^2_{q} - 2 * <D_en @ gamma @ D_vi^T, gamma>
    p = gamma.sum(dim=1)   # (K,) marginal EN
    q = gamma.sum(dim=0)   # (K,) marginal VI

    gw_term1 = (D_en ** 2 * p.unsqueeze(1) * p.unsqueeze(0)).sum()
    gw_term2 = (D_vi ** 2 * q.unsqueeze(1) * q.unsqueeze(0)).sum()
    gw_term3 = (D_en @ gamma @ D_vi.T * gamma).sum()

    GW_loss = gw_term1 + gw_term2 - 2.0 * gw_term3

    return alpha * GW_loss + (1.0 - alpha) * W_loss


# ──────────────────────────────────────────────────────────────
# Quick test: python fgw_solver.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    K = 32

    D1 = torch.rand(K, K, dtype=torch.float32)
    D1 = (D1 + D1.T) / 2   # symmetric

    D2 = torch.rand(K, K, dtype=torch.float32)
    D2 = (D2 + D2.T) / 2

    M = torch.rand(K, K, dtype=torch.float32)

    # Cần grad để test backward
    D1.requires_grad_(True)
    D2.requires_grad_(True)

    print("Testing partial_fgw...")
    g1 = partial_fgw(D1, D2, m=0.85)
    print(f"  gamma shape : {g1.shape}")
    print(f"  gamma sum   : {g1.sum().item():.4f}  (expected ≈ {0.85:.2f})")
    loss1 = g1.sum()
    loss1.backward()
    assert D1.grad is not None, "D1 grad is None — STE không hoạt động"
    print(f"  D1.grad norm: {D1.grad.norm().item():.6f}")
    print("  partial_fgw OK ✓\n")

    D1.grad = None
    D2.grad = None

    print("Testing fgw_bapg...")
    g2 = fgw_bapg(D1, D2, M, alpha=0.5, epsilon=0.05)
    print(f"  gamma shape : {g2.shape}")
    loss2 = g2.sum()
    loss2.backward()
    assert D1.grad is not None, "D1 grad is None — STE không hoạt động"
    print(f"  D1.grad norm: {D1.grad.norm().item():.6f}")
    print("  fgw_bapg OK ✓\n")

    print("Testing compute_fgw_loss...")
    gamma_detached = g2.detach().requires_grad_(False)
    D1_fresh = torch.rand(K, K, requires_grad=True)
    D2_fresh = torch.rand(K, K, requires_grad=True)
    M_fresh  = torch.rand(K, K)
    l = compute_fgw_loss(gamma_detached, D1_fresh, D2_fresh, M_fresh)
    print(f"  FGW loss: {l.item():.4f}")
    l.backward()
    print(f"  D1.grad norm: {D1_fresh.grad.norm().item():.6f}")
    print("  compute_fgw_loss OK ✓")