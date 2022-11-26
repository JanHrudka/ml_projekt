import matplotlib.pyplot as plt
import numpy as np


def correct_classification_overview(percentage_correct_mean_array,
                                    percentage_correct_std_array,
                                    model, selection_names):

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)

    indices = [i for i, _ in enumerate(selection_names)]
    colors_mask = np.logical_or(
        ((percentage_correct_mean_array -
          percentage_correct_std_array) > 0.5),
        ((percentage_correct_mean_array +
          percentage_correct_std_array) < 0.5)
    )
    colors = []
    for mask_value in colors_mask:

        if mask_value:

            colors.append("#4169E1")

        else:

            colors.append("#CCCCFF")

    ax.bar(indices, percentage_correct_mean_array,
           yerr=percentage_correct_std_array, color=colors)
    ax.plot([-1, len(selection_names)], [0.5, 0.5], color="red", linewidth=3,
            linestyle="--")
    ax.set_xticks(indices)
    ax.set_xticklabels(selection_names, rotation=45, ha="right")
    ax.set_title((f"Percentage of correctly classified frames\n"
                  f"{model}"))
    ax.set_ylim((0, 1))
    ax.set_xlim((-1, len(selection_names)))
    ax.grid()
