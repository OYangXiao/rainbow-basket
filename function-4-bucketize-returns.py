import math
import numpy as np

def calculate_friendly_step(min_val: float, max_val: float, num_bins: int) -> float:
    """
    计算友好的区间步长（如 0.01、0.05 等）
    
    参数:
        min_val: 最小值
        max_val: 最大值
        num_bins: 区间数量
    
    返回:
        调整后的步长（如 0.01、0.005 等）
    """
    raw_step = (max_val - min_val) / num_bins
    if raw_step == 0:
        return 0.0  # 处理极差为0的情况
    
    # 确定数量级（10的幂次）
    magnitude = 10 ** math.floor(math.log10(raw_step))
    
    # 将步长标准化到1~10之间，然后取最接近的1、2、5的倍数
    normalized_step = raw_step / magnitude
    if normalized_step < 1.5:
        friendly_step = 1 * magnitude
    elif normalized_step < 3:
        friendly_step = 2 * magnitude
    elif normalized_step < 7:
        friendly_step = 5 * magnitude
    else:
        friendly_step = 10 * magnitude
    
    return friendly_step

def bucketize_returns(name, returns, num_bins=16):
    """
    Bucketizes the returns into specified bins.

    Parameters:
    - returns: A list of returns.
    - bins: A list of bin edges.

    Returns:
    - A list of bucketized returns.
    """

    # Calculate the min and max of the returns
    min_val = min(returns)
    max_val = max(returns)
    # Create bins based on the min and max
    step = calculate_friendly_step(min_val, max_val, num_bins)

    adjusted_min = math.floor(min_val / step) * step
    adjusted_max = adjusted_min + step * num_bins

    if adjusted_max < max_val:
        adjusted_max += step

    # 生成区间边界（确保覆盖极值）
    bins = np.linspace(adjusted_min, adjusted_max, num_bins + 1)

    # print(f"bins: {bins}, len(bins): {len(bins)}")
    
    counts, _ = np.histogram(returns, bins=bins)

    data = {
        f"{bins[i]:.4f}~{bins[i+1]:.4f}": int(counts[i])
        for i in range(num_bins)
    }

    import importlib
    write_to_json = importlib.import_module("write-json").write_to_json
    # write daily log returns to file
    write_to_json(data, "output-4-bucketize-returns"+"-"+name.replace(" ", "-")+".json")

    return data



