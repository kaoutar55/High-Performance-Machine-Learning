"""Cost per completed task (Code 12.1)."""


def cost_per_task(gpu_price_h, n_gpus, tasks_per_s, utilization=0.6, other_cost_h=0.0):
    """tasks_per_s: goodput, the task rate at which the objectives hold.
    utilization: average load divided by goodput over the billing period."""
    hourly = gpu_price_h * n_gpus + other_cost_h
    tasks_per_h = tasks_per_s * 3600 * utilization
    return hourly / tasks_per_h


if __name__ == "__main__":
    # one GPU at 0.5 tasks/s goodput; a price of 1 per GPU-hour gives GPU-hours,
    # times 3,600 gives GPU-seconds per task. Multiply GPU-hours by your own price.
    print(f"{3600 * cost_per_task(1, 1, 0.5, utilization=1.0):.1f}")   # 2.0
    print(f"{3600 * cost_per_task(1, 1, 0.5, utilization=0.4):.1f}")   # 5.0
