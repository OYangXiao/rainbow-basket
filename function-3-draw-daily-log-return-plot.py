
# draw a line chart for each index
import matplotlib.pyplot as plt

plot_config = {
    "SSE Composite Index": {
        "color": "red",
    },
    "CSI 300 Index": {
        "color": "blue",
    },
    "CSI 500 Index": {
        "color": "green",
    },
}

def draw_daily_log_returns_plot(daily_log_returns):
    """
    Draw daily log returns plot
    :param daily_log_returns: daily log returns data
    :return: None
    """
    # Plot each index's daily log returns
    for index_name in daily_log_returns:
        # Set the figure size
        plt.figure(figsize=(12, 6))

        plt.plot(
            daily_log_returns[index_name],
            color=plot_config[index_name]["color"],
            label=index_name,
        )
        plt.axhline(0, color="black", lw=0.5, ls="--")
        plt.title("Daily Log Returns - " + index_name)
        plt.xlabel("Days")
        plt.ylabel("Daily Log Returns")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = "output-3-daily-log-returns-plot-"+index_name.replace(" ", "-")+".png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-resolution PNG
        plt.close()
