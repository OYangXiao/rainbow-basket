import numpy as np

def draw_histogram_returns(index_name, bucketized_data):
    """
    Draws a histogram of the daily log returns for a given index.

    Parameters:
    - index_name: The name of the index.
    - bucketized_data: A list of bucketized returns for the index.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt

    bin_ranges = list(bucketized_data.keys())
    counts = list(bucketized_data.values())
    bin_centers = [np.mean([float(x.split('~')[0]), float(x.split('~')[1])]) for x in bin_ranges]
    bin_width = bin_centers[1] - bin_centers[0]  # 计算柱宽
    
    # 创建直方图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bin_centers, counts, width=bin_width*0.9, 
                   edgecolor='black', linewidth=0.5,
                   color='skyblue', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 优化图表
    plt.title('Rainbow Basket Option Returns Distribution - '+ index_name, fontsize=14, pad=20)
    plt.xlabel('Return Range', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(bin_centers, [f'{x:.3f}' for x in bin_centers], rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    filename = "output-5-histogram-returns-plot-"+index_name.replace(" ", "-")+".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-resolution PNG
    plt.close()
