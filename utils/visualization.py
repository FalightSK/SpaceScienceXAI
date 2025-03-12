import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(single_batch_attention, feature_names, figsize=(10, 8)):
    plt.figure(figsize=figsize)  # Adjust figure size for better readability
    sns.heatmap(single_batch_attention,
                annot=True,  # Show the values in the cells
                cmap="viridis",  # Choose a colormap
                xticklabels=feature_names,  # Use the feature names
                yticklabels=feature_names,  # Use the feature names
                cbar_kws={'label': 'Attention Weight'})  # Add a colorbar label
    plt.xlabel("Key Features")
    plt.ylabel("Query Features")
    plt.title("Transformer Attention Weights")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

def context_analysis(weights, out_feature_idx, figsize=(12, 6)):
    # Separate feature and context weights
    context_weights = weights[:, out_feature_idx:]  # Weights for context embedding

    print("Context Weights Shape:", context_weights.shape)  # (embedding_dim, context_dim)

    # Analyze context_weights:
    #   - Option 1: Average absolute weights per context dimension
    avg_context_weights = np.mean(np.abs(context_weights), axis=0)
    # Sort to easily see most important
    sorted_indices = np.argsort(avg_context_weights)[::-1]  # Descending order

    print("Average Absolute Context Weights (Sorted):")
    for i in sorted_indices:
        print(f"  Context Dimension {i}: {avg_context_weights[i]:.4f}")

    #  - Option 2: Visualize as a heatmap (if context_dim isn't too large)
    if context_weights.shape[1] < 128: # Arbitrary limit, adjust as needed
        plt.figure(figsize=figsize)
        sns.heatmap(context_weights, annot=False, cmap="viridis",
                    xticklabels=[f"Context_{i}" for i in range(context_weights.shape[1])],
                    yticklabels=[f"Output_{i}" for i in range(context_weights.shape[0])],
                    cbar_kws={'label': 'Weight'})
        plt.xlabel("Context Embedding Dimensions")
        plt.ylabel("Output Embedding Dimensions")
        plt.title("Context Integration Weights")
        plt.tight_layout()
        plt.show()