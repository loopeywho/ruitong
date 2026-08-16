# Habr / VC.ru — Ascend vs CUDA: Кросс-платформенная эквивалентность LLM

**Заголовок:** Ascend 910B vs NVIDIA A40: всего 15% совпадения top-1 токенов. Почему это важно для индустрии.

Мы запустили проект [Ruitong](https://ruitong.io) — открытый бенчмарк, измеряющий эквивалентность LLM на разном «железе» (CUDA, Ascend, AMD).

## Результаты

Запустили Qwen3-8B на Ascend (DashScope) и A40 (CUDA) с идентичными параметрами:
- 61 промпт, temperature=0, seed=1234
- Каждый промпт прогнали 3 раза (один холодный — отбросили)

| Метрика | Значение |
|---------|----------|
| Совпадение top-1 токена | **15.02%** |
| Промптов с расхождением | **60/61 (98.4%)** |
| Top-5 пересечение | **20.92%** |

Для сравнения: A40 vs RTX 6000 Ada (та же CUDA) — ~19% расхождений. A40 vs Ascend — **98%**. Это на порядок больше.

## Почему это важно

CANN-стек Ascend вносит существенные численные различия в forward pass. Если вы используете:
- Logit-based alignment
- RLHF
- Speculative decoding
- Любую технику, зависящую от распределения токенов

...результаты могут НЕ переноситься между CUDA и Ascend.

## Открытые данные

Все корпуса и код открыты:
📊 [ruitong.io](https://ruitong.io)
🐙 [github.com/loopeywho/ruitong](https://github.com/loopeywho/ruitong)

Следующая цель: AMD MI350 vs CUDA.

---

*Ruitong Project — measuring cross-accelerator LLM equivalence since 2026.*