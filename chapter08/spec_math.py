"""Expected tokens per target pass and modeled speedup of speculative decoding (Leviathan et al., 2023).

    python spec_math.py
"""


def expected_tokens(alpha, gamma):
    return gamma + 1 if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):                 # c: cost of one draft step / cost of one target step
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


if __name__ == "__main__":
    for alpha in (0.6, 0.8, 0.9):
        best = max(range(1, 17), key=lambda g: speedup(alpha, g, 0.05))
        print(f"alpha={alpha}: E(gamma=4)={expected_tokens(alpha, 4):.2f} "
              f"speedup(gamma=4)={speedup(alpha, 4, 0.05):.2f}x "
              f"best gamma={best} ({speedup(alpha, best, 0.05):.2f}x)")
