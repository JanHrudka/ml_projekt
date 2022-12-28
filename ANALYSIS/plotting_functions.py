import numpy as np
import matplotlib.pyplot as plt


def correct_classification_overview(mean_array, std_array, model,
                                    selection_names):

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)

    indices = [i for i, _ in enumerate(selection_names)]
    colors_mask = np.logical_or(
        ((mean_array -
          std_array) > 0.5),
        ((mean_array +
          std_array) < 0.5)
    )
    colors = []
    for mask_value in colors_mask:

        if mask_value:

            colors.append("#4169E1")

        else:

            colors.append("#CCCCFF")

    ax.bar(indices, mean_array, yerr=std_array, color=colors, alpha=0.95)
    ax.plot([-1, len(selection_names)], [0.5, 0.5], color="red", linewidth=3,
            linestyle="--", zorder=0)
    xtick_label = [
        f"{i:.2f} \u00b1 {j:.2f}\n\n{k}" for i, j, k in zip(mean_array,
                                                            std_array,
                                                            selection_names)
    ]
    ax.set_xticks(indices)
    ax.set_xticklabels(xtick_label, fontsize=15, fontweight="bold", y=0.085)
    ax.set_title(f"{model}", fontsize=14, fontweight="bold")
    ax.set_ylabel(("Ratio of correctly classified frames"),
                  fontsize=14, fontweight="bold")
    ax.set_ylim((0, 1))
    plt.yticks(fontsize=16)
    ax.set_xlim((-1, len(selection_names)))
    ax.yaxis.grid(color='gray', linestyle='dashed')
