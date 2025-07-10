import matplotlib.pyplot as plt
from utils_display import nice_colorbar


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
