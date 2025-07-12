import numpy as np
import matplotlib.pyplot as plt
from utils_display import nice_colorbar
import seaborn as sns
sns.reset_orig()


def display_scaled_dot_product(q, k, v, mask, weighted_sum_of_values, attention_weights) -> None:

    fig, ax = plt.subplots(2, 3)
    fig.set_dpi(300)
    fig.set_size_inches(16, 10, forward=True)

    axx = ax[0, 0]
    fig.sca(axx)
    im = plt.imshow(q)
    plt.title("Q {}".format(q.shape))
    plt.xlabel("Embedding dimensionality")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    axx = ax[0, 1]
    fig.sca(axx)
    im = plt.imshow(k)
    plt.title("K {}".format(k.shape))
    plt.xlabel("Embedding dimensionality")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    axx = ax[0, 2]
    fig.sca(axx)
    im = plt.imshow(v)
    plt.title("V {}".format(v.shape))
    plt.xlabel("Embedding dimensionality")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    axx = ax[1, 0]
    fig.sca(axx)
    im = plt.imshow(mask, cmap="gray")
    plt.title("Mask {}".format(mask.shape))
    plt.xlabel("Sequence length")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    axx = ax[1, 1]
    fig.sca(axx)
    im = plt.imshow(attention_weights)
    plt.title("Attention weights {}".format(attention_weights.shape))
    plt.xlabel("Sequence length")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    axx = ax[1, 2]
    fig.sca(axx)
    im = plt.imshow(weighted_sum_of_values)
    plt.title("Weighted sum of values {}".format(weighted_sum_of_values.shape))
    plt.xlabel("Embedding dimensionality")
    plt.ylabel("Sequence length")
    nice_colorbar(im, axx)

    plt.show()
    
    
def display_positional_encoding(positional_encoding) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
    pos = ax.imshow(
        positional_encoding, cmap="RdGy",      
        extent=(1,positional_encoding.shape[1]+1,positional_encoding.shape[0]+1,1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1]+[i*10 for i in range(1,1+positional_encoding.shape[1]//10)])
    ax.set_yticks([1]+[i*10 for i in range(1,1+positional_encoding.shape[0]//10)])
    plt.show()
    
    
def display_positional_encoding_profiles(positional_encoding) -> None:
    sns.set_theme()
    fig, ax = plt.subplots(2, 2, figsize=(12,4))
    ax = [a for a_list in ax for a in a_list]
    for i in range(len(ax)):
        ax[i].plot(
            np.arange(1,17), positional_encoding[i,:16], color=f'C{i}', marker="o", markersize=6,
            markeredgecolor="black")
        ax[i].set_title(f"Encoding in hidden dimension {i+1}")
        ax[i].set_xlabel("Position in sequence", fontsize=10)
        ax[i].set_ylabel("Positional encoding", fontsize=10)
        ax[i].set_xticks(np.arange(1,17))
        ax[i].tick_params(axis='both', which='major', labelsize=10)
        ax[i].tick_params(axis='both', which='minor', labelsize=8)
        ax[i].set_ylim(-1.2, 1.2)
    fig.subplots_adjust(hspace=0.8)
    sns.reset_orig()
    plt.show()
    

def display_lr_scheduler(lr_scheduler) -> None:
    epochs = list(range(2000))
    sns.set()
    plt.figure(figsize=(8,3))
    plt.plot(epochs, [lr_scheduler(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title("Cosine Warm-up Learning Rate Scheduler")
    plt.show()
    sns.reset_orig()    
