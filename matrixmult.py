import time
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def benchmark_matrix_mult(size, device, iterations=3):
    torch.manual_seed(0)
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warm-up
    for _ in range(2):
        _ = torch.matmul(a, b).sum()
        synchronize(device)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = torch.matmul(a, b)
        _ = out.sum().item()
        synchronize(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    flops = 2 * size**3
    gflops = flops / 1e9 / avg_time
    return avg_time, gflops


def main():
    # Hardcoded parameters (no CLI parameters)
    size = 2048
    iterations = 3

    # # Prefer CUDA, else MPS, else exit
    # if torch.cuda.is_available():
    #     name, device = "cuda", torch.device("cuda")
    # elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    #     name, device = "mps", torch.device("mps")
    #run cpu
    name, device = "cpu", torch.device("cpu")
    # else:
    #     print("No supported GPU backend available (CUDA or MPS).")
    #     return

    print(f"Benchmarking {size}x{size} matrix multiplication on {name}")
    avg_time, gflops = benchmark_matrix_mult(size, device, iterations)
    print(f"{name}: average time = {avg_time:.4f}s, performance = {gflops:.2f} GFLOPS")


if __name__ == "__main__":
    main()
