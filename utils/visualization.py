# deep_image_analyzer/utils/visualization.py

# Import plotting library
import matplotlib.pyplot as plt

def plot_confidence(probs, labels, top_k=5):
    """
    Plot the top_k class probabilities as a horizontal bar chart.
    """
    top = probs.argsort()[-top_k:][::-1]  # Indices of top_k probabilities
    fig, ax = plt.subplots(figsize=(6,4))
    # Draw bars
    ax.barh(range(top_k), probs[top], color="skyblue")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([labels[i] for i in top])
    ax.set_xlabel("Probability")
    ax.set_title("Top Predictions")
    fig.tight_layout()
    return fig
