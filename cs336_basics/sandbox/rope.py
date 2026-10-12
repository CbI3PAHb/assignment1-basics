import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("Размерность 'dim' должна быть четной.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        # Конкатенируем в порядке (-x2, x1), что эквивалентно созданию [-y, x] для каждой пары [x, y]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch, num_heads, seq_len, head_dim)
        _, _, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Длина последовательности ({seq_len}) превышает максимальную предвычисленную длину ({self.max_seq_len})."
            )

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 2. Изменяем форму cos/sin для корректного broadcasting'а с x.
        # Из [seq_len, dim] делаем [1, 1, seq_len, dim] чтобы соответствовать [B, H, L, D_head]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_x = x * cos + self._rotate_half(x) * sin
        return rotated_x


# --- Пример использования ---
if __name__ == "__main__":
    batch_size = 1  # Упростим до 1 для наглядности
    num_heads = 1
    d_head = 64
    max_test_len = 30
    rope = RotaryPositionalEmbeddings(dim=d_head, max_seq_len=128)
    q_base_vec = torch.randn(batch_size, num_heads, 1, d_head)
    k_base_vec = torch.randn(batch_size, num_heads, 1, d_head)

    q_sequence = q_base_vec.repeat(1, 1, max_test_len, 1)
    k_sequence = k_base_vec.repeat(1, 1, max_test_len, 1)
    q_rotated_all = rope(q_sequence)
    k_rotated_all = rope(k_sequence)
    q_m1 = q_rotated_all[:, :, 5, :]
    k_n1 = k_rotated_all[:, :, 7, :]
    dot_prod_1 = torch.sum(q_m1 * k_n1)

    q_m2 = q_rotated_all[:, :, 15, :]
    k_n2 = k_rotated_all[:, :, 17, :]
    dot_prod_2 = torch.sum(q_m2 * k_n2)

    q_m3 = q_rotated_all[:, :, 8, :]
    k_n3 = k_rotated_all[:, :, 13, :]
    dot_prod_3 = torch.sum(q_m3 * k_n3)

    q_m4 = q_rotated_all[:, :, 10, :]
    k_n4 = k_rotated_all[:, :, 15, :]
    dot_prod_4 = torch.sum(q_m4 * k_n4)

    assert torch.allclose(
        dot_prod_1, dot_prod_2
    ), "Результаты для расстояния 2 должны быть одинаковыми!"
    assert torch.allclose(
        dot_prod_3, dot_prod_4
    ), "Результаты для расстояния 5 должны быть одинаковыми!"
