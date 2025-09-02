import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    """
    Реализация Rotary Positional Embeddings (RoPE).
    
    Аргументы:
        dim (int): Размерность эмбеддингов, к которым применяется RoPE. Должна быть четной.
        max_seq_len (int): Максимальная длина последовательности, для которой будут предвычислены эмбеддинги.
        base (int): Константа "тета" (Θ) из формулы.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Размерность 'dim' должна быть четной.")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 1. Вычисляем частоты (inv_freq)
        # Формула: 1 / (base^(2k/d)) для k in [0, 1, ..., d/2 - 1]
        # inv_freq будет иметь форму [dim / 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        # 2. Создаем тензор позиций
        # t будет иметь форму [max_seq_len]
        t = torch.arange(self.max_seq_len, dtype=torch.float32)

        # 3. Вычисляем углы (freqs) путем внешнего произведения позиций на частоты
        # freqs будет иметь форму [max_seq_len, dim / 2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # 4. Дублируем частоты для каждой пары (x, y)
        # emb будет иметь форму [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 5. Регистрируем cos и sin как буферы
        # persistent=False означает, что эти тензоры являются частью состояния модуля,
        # но не будут сохраняться в state_dict. Они легко пересчитываются.
        # Форма: [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Вспомогательная функция для эффективного поворота."""
        # Разделяем тензор на две половины по последней оси
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        # Конкатенируем в порядке (-x2, x1), что эквивалентно созданию [-y, x] для каждой пары [x, y]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет RoPE к входному тензору.
        
        Аргументы:
            x (torch.Tensor): Входной тензор, например, query или key.
                              Ожидаемая форма: [..., seq_len, dim], например, [B, H, L, D_head].
                              где D_head - это self.dim.
        
        Возвращает:
            torch.Tensor: Тензор той же формы, что и x, с примененными позиционными вращениями.
        """
        # Получаем реальную длину последовательности из входного тензора
        # x.shape = (batch, num_heads, seq_len, head_dim)
        _, _, seq_len, _ = x.shape
        
        if seq_len > self.max_seq_len:
             raise ValueError(
                f"Длина последовательности ({seq_len}) превышает максимальную предвычисленную длину ({self.max_seq_len})."
             )

        # 1. Берем предвычисленные значения cos и sin для нужной длины последовательности
        # cos/sin будут иметь форму [seq_len, dim]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # 2. Изменяем форму cos/sin для корректного broadcasting'а с x.
        # Из [seq_len, dim] делаем [1, 1, seq_len, dim] чтобы соответствовать [B, H, L, D_head]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 3. Применяем формулу вращения: x_rot = x * cos + rotate_half(x) * sin
        # rotate_half(x) создает тензор, где для каждой пары (x_i, y_i) получается (-y_i, x_i)
        # x * cos -> (x_i * cos, y_i * cos)
        # rotate_half(x) * sin -> (-y_i * sin, x_i * sin)
        # Сумма: (x_i * cos - y_i * sin, y_i * cos + x_i * sin) - это и есть наша формула!
        rotated_x = x * cos + self._rotate_half(x) * sin
        
        return rotated_x

# --- Пример использования ---
if __name__ == "__main__":
    batch_size = 1 # Упростим до 1 для наглядности
    num_heads = 1
    d_head = 64
    max_test_len = 30
    
    # Создаем RoPE модуль
    rope = RotaryPositionalEmbeddings(dim=d_head, max_seq_len=128)
    
    # --- КОРРЕКТНАЯ ДЕМОНСТРАЦИЯ СВОЙСТВА RoPE ---

    # 1. Создаем два ОДИНАКОВЫХ базовых вектора, которые мы будем вращать.
    q_base_vec = torch.randn(batch_size, num_heads, 1, d_head)
    k_base_vec = torch.randn(batch_size, num_heads, 1, d_head)

    # 2. Создаем "фиктивные" последовательности, повторяя базовые векторы.
    # Это нужно, чтобы наш RoPE модуль мог применить вращения для разных позиций.
    q_sequence = q_base_vec.repeat(1, 1, max_test_len, 1) # Тензор из 30 одинаковых q_base_vec
    k_sequence = k_base_vec.repeat(1, 1, max_test_len, 1) # Тензор из 30 одинаковых k_base_vec

    # 3. Применяем RoPE ко всей последовательности.
    # Теперь q_rotated_all[..., i, :] содержит q_base_vec, повернутый для позиции i.
    q_rotated_all = rope(q_sequence)
    k_rotated_all = rope(k_sequence)
    
    # 4. Проверяем скалярное произведение для пар с одинаковым относительным расстоянием.
    
    # --- Пара 1: позиции m=5, n=7 (расстояние n-m = 2) ---
    q_m1 = q_rotated_all[:, :, 5, :]  # q_base_vec повернут для позиции 5
    k_n1 = k_rotated_all[:, :, 7, :]  # k_base_vec повернут для позиции 7
    dot_prod_1 = torch.sum(q_m1 * k_n1)

    # --- Пара 2: позиции m=15, n=17 (расстояние n-m = 2) ---
    q_m2 = q_rotated_all[:, :, 15, :] # q_base_vec повернут для позиции 15
    k_n2 = k_rotated_all[:, :, 17, :] # k_base_vec повернут для позиции 17
    dot_prod_2 = torch.sum(q_m2 * k_n2)

    # --- Пара 3: позиции m=8, n=13 (расстояние n-m = 5) ---
    q_m3 = q_rotated_all[:, :, 8, :]
    k_n3 = k_rotated_all[:, :, 13, :]
    dot_prod_3 = torch.sum(q_m3 * k_n3)
    
    # --- Пара 4: позиции m=10, n=15 (расстояние n-m = 5) ---
    q_m4 = q_rotated_all[:, :, 10, :]
    k_n4 = k_rotated_all[:, :, 15, :]
    dot_prod_4 = torch.sum(q_m4 * k_n4)

    print("\n--- Проверка свойства RoPE ---")
    print(f"Расстояние 2 (pos 5, 7):   {dot_prod_1.item():.4f}")
    print(f"Расстояние 2 (pos 15, 17): {dot_prod_2.item():.4f}")
    assert torch.allclose(dot_prod_1, dot_prod_2), "Результаты для расстояния 2 должны быть одинаковыми!"
    print("-> Проверка для расстояния 2 пройдена.\n")

    print(f"Расстояние 5 (pos 8, 13):  {dot_prod_3.item():.4f}")
    print(f"Расстояние 5 (pos 10, 15): {dot_prod_4.item():.4f}")
    assert torch.allclose(dot_prod_3, dot_prod_4), "Результаты для расстояния 5 должны быть одинаковыми!"
    print("-> Проверка для расстояния 5 пройдена.\n")
