import chess_agent.model as model
import matplotlib.pyplot as plt
import numpy as np


def plot_layer(layer):

    weights = layer.get_weights()
    w = weights[0]  # weights
    b = weights[1]  # biases

    filters = w.shape[3]
    channels = w.shape[2]

    fig = plt.figure(figsize=(channels, filters))

    plt_ix = 1

    for filter_ix in range(filters):
        f_w = w[..., filter_ix]
        f_b = w[..., filter_ix]

        for channel in range(channels):
            ax = fig.add_subplot(filters, channels, plt_ix)
            plt_ix += 1
            plt.imshow(f_w[..., channel], cmap='gray', interpolation="none")
            #ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()


m = model.get_model()
l = m.get_layer(name='conv_1')
plot_layer(l)
