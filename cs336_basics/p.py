"""Учебный GPTQ для одного Linear; только фиксированные INT4-сетки.

Python >= 3.10, PyTorch. Запуск: python gptq_walkthrough.py
CPU/FP64 используется для проверок алгебры, FP32 — для рабочего примера.
Это не production-квантайзер: нет packing, групповых scales, act-order и CUDA kernels.
Одна шкала на выходную строку; запускать вне autocast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class Step:
    j: int
    before: Tensor
    after: Tensor
    q: Tensor
    r: Tensor
    inverse_before: Tensor
    predicted_increase: float


@torch.no_grad()
def prepare_problem(
    X: Tensor,
    W: Tensor,
    *,
    percdamp: float = 0.01,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Return Wc, H_work, scale, dead, damping; inputs are not mutated.

    H_raw = 2 X.T X / N. Wc is W with dead columns zeroed.
    H_work has isolated unit diagonals for dead coordinates and fixed damping.
    No adaptive damping is hidden in this function.
    """
    if X.ndim != 2 or W.ndim != 2 or X.shape[1] != W.shape[1]:
        raise ValueError("Expected X[N, d_in], W[d_out, d_in].")
    if min(*X.shape, *W.shape) < 1:
        raise ValueError("N, d_in, d_out must be positive.")
    if X.device != W.device or not (X.is_floating_point() and W.is_floating_point()):
        raise ValueError("X and W must be floating-point tensors on the same device.")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use FP32 for work or FP64 for algebra checks.")
    if not math.isfinite(percdamp) or percdamp < 0:
        raise ValueError("percdamp must be finite and nonnegative.")
    if not bool(torch.isfinite(X).all() and torch.isfinite(W).all()):
        raise ValueError("Inputs contain NaN or Inf.")

    x = X.to(dtype=dtype)
    Wc = W.to(dtype=dtype).clone()
    # Фиксируем сетку до первого изменения весов.
    scale = Wc.abs().amax(dim=1) / 7
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    H = (2.0 / X.shape[0]) * (x.T @ x)
    H = (H + H.T) / 2  # Убираем возможную вычислительную асимметрию.
    if not bool(torch.isfinite(H).all()):
        raise ValueError("Gram matrix overflowed the selected work dtype.")
    dead = H.diagonal() == 0
    H.masked_fill_(dead[:, None], 0)
    H.masked_fill_(dead[None, :], 0)
    H.diagonal().masked_fill_(dead, 1)
    Wc.masked_fill_(dead[None, :], 0)

    damping = percdamp * H.diagonal().mean().item()
    H.diagonal().add_(damping)
    return Wc, H, scale, dead, damping


@torch.no_grad()
def inverse_cholesky_flip(H: Tensor) -> Tensor:
    """Return upper U with U.T @ U = H^{-1}; H must already be SPD."""
    PHP = H.flip((0, 1))
    try:
        L = torch.linalg.cholesky(PHP, upper=False)
    except RuntimeError as exc:
        raise ValueError("Cholesky failed: inspect H and increase damping before retrying.") from exc
    I = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    M = torch.linalg.solve_triangular(L, I, upper=False)
    return M.flip((0, 1))


def quantize_column(v: Tensor, scale: Tensor) -> Tensor:
    """[d_out] -> [d_out]: dequantized INT4 values, not packed integer codes."""
    return (v / scale).round().clamp(-8, 7) * scale


@torch.no_grad()
def quantize_with_inverse(
    Wc: Tensor,
    H: Tensor,
    scale: Tensor,
    *,
    capture: bool = False,
) -> tuple[Tensor, list[Step]]:
    """Учебный цикл: обновляет веса И текущую обратную матрицу после шага."""
    V = Wc.clone()
    Q = torch.empty_like(V)
    B_current = torch.linalg.inv(H)
    trace: list[Step] = []
    d_in = V.shape[1]

    for j in range(d_in):
        before = V.clone() if capture else None
        q = quantize_column(V[:, j], scale)
        r = V[:, j] - q
        Q[:, j] = q

        # Локальная координата 0 в B_current = исходная координата j.
        V[:, j:] -= (r / B_current[0, 0])[:, None] * B_current[0, :][None, :]
        V[:, j] = q  # Точная фиксация выбранного значения в floating-point.

        if capture:
            trace.append(
                Step(
                    j=j,
                    before=before,
                    after=V.clone(),
                    q=q.clone(),
                    r=r.clone(),
                    inverse_before=B_current.clone(),
                    predicted_increase=(r.square().sum() / (2 * B_current[0, 0])).item(),
                )
            )

        # Это действие готовит геометрию СЛЕДУЮЩЕГО шага.
        if j + 1 < d_in:
            B_updated = B_current - torch.outer(B_current[:, 0], B_current[0, :]) / B_current[0, 0]
            B_current = B_updated[1:, 1:]

    return Q, trace


@torch.no_grad()
def quantize_with_u(Wc: Tensor, U: Tensor, scale: Tensor) -> Tensor:
    """Те же последовательные поправки; U фиксирована, B_current не нужна."""
    V = Wc.clone()
    Q = torch.empty_like(V)
    for j in range(V.shape[1]):
        q = quantize_column(V[:, j], scale)
        r = V[:, j] - q
        err = r / U[j, j]
        Q[:, j] = q
        V[:, j:] -= err[:, None] * U[j, j:][None, :]
        V[:, j] = q
    return Q


@torch.no_grad()
def quantize_blocked(
    Wc: Tensor,
    U: Tensor,
    scale: Tensor,
    blocksize: int = 128,
) -> Tensor:
    """Последовательные решения внутри блока, отложенный matmul за блоком."""
    if blocksize < 1:
        raise ValueError("blocksize must be positive.")
    V = Wc.clone()
    Q = torch.empty_like(V)
    d_in = V.shape[1]
    for a in range(0, d_in, blocksize):
        b = min(a + blocksize, d_in)
        Vb = V[:, a:b].clone()
        Err = torch.empty_like(Vb)
        for j in range(a, b):
            k = j - a
            q = quantize_column(Vb[:, k], scale)
            err = (Vb[:, k] - q) / U[j, j]
            Q[:, j] = q
            Err[:, k] = err
            Vb[:, k:] -= err[:, None] * U[j, j:b][None, :]
            Vb[:, k] = q
        # V[:, a:b] после обработки не используется: результат находится в Q.
        V[:, b:] -= Err @ U[a:b, b:]
    return Q


def quadratic_loss(V: Tensor, Wc: Tensor, H: Tensor) -> Tensor:
    D = V - Wc
    return 0.5 * ((D @ H) * D).sum()


def output_loss(X: Tensor, W: Tensor, Q: Tensor) -> Tensor:
    # Суммируем по выходам и токенам, делим только на N, не на d_out.
    return (X @ (W - Q).T).square().sum() / X.shape[0]


@torch.no_grad()
def run_checks() -> dict[str, float | int | str]:
    torch.set_num_threads(1)
    checks = 0
    max_residual = 0.0

    def close(actual: Tensor, expected: Tensor, *, tol: float = 2e-9) -> None:
        nonlocal checks, max_residual
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
        checks += 1
        if actual.numel():
            max_residual = max(max_residual, (actual - expected).abs().max().item())

    # Один сквозной пример из заметки, без damping и с целой сеткой.
    X = torch.tensor([[1.0, 0.0], [-2.0, 1.0]], dtype=torch.float64)
    W = torch.tensor([[0.8, 0.4]], dtype=torch.float64)
    H = 2 * (X.T @ X) / X.shape[0]
    scale = torch.ones(1, dtype=torch.float64)
    U = inverse_cholesky_flip(H)
    Q, trace = quantize_with_inverse(W, H, scale, capture=True)
    close(H, torch.tensor([[5.0, -2.0], [-2.0, 1.0]], dtype=H.dtype))
    close(U, torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=H.dtype))
    close(trace[0].after, torch.tensor([[1.0, 0.8]], dtype=H.dtype))
    close(Q, torch.tensor([[1.0, 1.0]], dtype=H.dtype))
    close(output_loss(X, W, Q), H.new_tensor(0.04))
    Q_rtn = (W / scale[:, None]).round().clamp(-8, 7) * scale[:, None]
    close(output_loss(X, W, Q_rtn), H.new_tensor(0.34))

    problems = 0
    for seed in range(4):
        gen = torch.Generator().manual_seed(seed)
        # Включая N < d_in, остаточные блоки, d_in=1 и dead-каналы.
        for N, d_in, d_out, make_dead in [
            (2, 4, 3, False),
            (7, 3, 2, False),
            (8, 7, 4, True),
            (5, 1, 2, False),
            (3, 4, 2, True),
        ]:
            X = torch.randn(N, d_in, generator=gen, dtype=torch.float64)
            W = torch.randn(d_out, d_in, generator=gen, dtype=torch.float64)
            if make_dead:
                X[:, 0] = 0
            Wc, H, scale, dead, damping = prepare_problem(X, W, dtype=torch.float64, percdamp=0.03)
            B = torch.linalg.inv(H)
            U = inverse_cholesky_flip(H)
            close(U.T @ U, B)
            close(U, torch.linalg.cholesky(B, upper=True))
            close(X @ Wc.T, X @ W.T)

            Q_b, trace = quantize_with_inverse(Wc, H, scale, capture=True)
            close(Q_b, quantize_with_u(Wc, U, scale))
            for blocksize in (1, 2, 3, d_in + 1):
                close(Q_b, quantize_blocked(Wc, U, scale, blocksize))
            if bool(dead.any()):
                close(Q_b[:, dead], torch.zeros_like(Q_b[:, dead]))

            predicted_total = 0.0
            for step in trace:
                j = step.j
                Bj = torch.linalg.inv(H[j:, j:])
                close(step.inverse_before, Bj)
                Ut = U[j:, j:]
                close(Ut.T @ Ut, Bj)
                close(U[j, j:].reshape(1, -1), Bj[0:1, :] / Bj[0, 0].sqrt())
                close(U[j, j].square(), Bj[0, 0])

                # После компенсации оставшиеся веса — условный минимум.
                grad = (step.after - Wc) @ H
                close(grad[:, j + 1 :], torch.zeros_like(grad[:, j + 1 :]))
                close(step.after[:, :j], step.before[:, :j])
                observed = quadratic_loss(step.after, Wc, H) - quadratic_loss(step.before, Wc, H)
                close(observed, H.new_tensor(step.predicted_increase))
                predicted_total += step.predicted_increase

            close(quadratic_loss(Q_b, Wc, H), H.new_tensor(predicted_total))
            # Dead-координаты Q и Wc нулевые, поэтому искусственный dead-блок не вносит loss.
            close(
                quadratic_loss(Q_b, Wc, H),
                output_loss(X, W, Q_b) + 0.5 * damping * (Q_b - Wc).square().sum(),
            )

            # Удаление ЛЮБОЙ одной координаты из одной и той же исходной матрицы.
            for k in range(d_in):
                keep = torch.arange(d_in) != k
                updated = B - torch.outer(B[:, k], B[k, :]) / B[k, k]
                close(updated[keep][:, keep], torch.linalg.inv(H[keep][:, keep]))
            problems += 1

    # Все калибровочные активации равны нулю.
    X = torch.zeros(2, 3, dtype=torch.float64)
    W = torch.tensor([[0.2, -0.6, 1.3]], dtype=torch.float64)
    Wc, H, scale, dead, _ = prepare_problem(X, W, dtype=torch.float64)
    Q = quantize_blocked(Wc, inverse_cholesky_flip(H), scale, blocksize=2)
    close(Q, torch.zeros_like(Q))
    close(output_loss(X, W, Q), H.new_tensor(0.0))

    # BF16-входы -> рабочие FP32-матрицы -> BF16-результат.
    gen = torch.Generator().manual_seed(2026)
    X = torch.randn(20, 5, generator=gen).to(torch.bfloat16)
    W = torch.randn(3, 5, generator=gen).to(torch.bfloat16)
    Wc, H, scale, _, _ = prepare_problem(X, W, dtype=torch.float32)
    Q32 = quantize_blocked(Wc, inverse_cholesky_flip(H), scale, blocksize=3)
    Qbf16 = Q32.to(W.dtype)
    assert Wc.dtype == H.dtype == scale.dtype == torch.float32
    assert Qbf16.dtype == torch.bfloat16 and Qbf16.shape == W.shape
    assert bool(torch.isfinite(Qbf16).all())
    checks += 3

    return {
        "torch_version": torch.__version__,
        "random_problems": problems,
        "checks_passed": checks,
        "max_fp64_absolute_residual": max_residual,
        "example_gptq_loss": 0.04,
        "example_rtn_loss": 0.34,
    }


if __name__ == "__main__":
    import json

    report = run_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2))
