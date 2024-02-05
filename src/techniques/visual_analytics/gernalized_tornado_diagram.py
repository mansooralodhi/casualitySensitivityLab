import matplotlib.pyplot as plt
import numpy as np
from typing import List


# todo: plot bi-direction tonrado diagram to plot lower and upper bound in same digram

class GeneralizedTornadoDiagram(object):
    def __init__(self, variables_names, preview=True, save_fig=False, filename=None):
        self.preview = preview
        self.save_fig = save_fig
        self.filename = filename
        self.variables_names = variables_names

        self.default_figsize = (8, 4)
        self.default_height = 0.35
        self.default_color = "skyblue"
        self.default_x_lablel = "Features Impact"
        self.default_title = "Sensitivity Analysis of IoI"

        self.individual_sensitivity_color = 'green'
        self.interaction_sensitivity_color = 'yellow'
        self.total_sensitivity_color = 'blue'
        self.bar_positions = np.arange(len(self.variables_names))

    def configure_plot(self):
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel(self.default_x_lablel)
        plt.title(self.default_title)
        plt.legend(loc='lower right')
        plt.tight_layout()

    def plot_sensitivities(self, individual_sensitivity: List[float],
                         interaction_sensitivity: List[float],
                         total_sensitivity: List[float],
                           title="",
                           savePath=""):

        assert len(individual_sensitivity) == len(interaction_sensitivity) == len(total_sensitivity) == len(self.variables_names), 'Sensitivities Not Synchronized.'
        plt.figure(figsize=self.default_figsize)
        plt.barh(self.variables_names, total_sensitivity,
                 color=self.total_sensitivity_color, label='Total', height=self.default_height)
        plt.barh(self.variables_names, interaction_sensitivity, left=total_sensitivity,
                 color=self.interaction_sensitivity_color, label='Interaction', height=self.default_height)
        plt.barh(self.variables_names, individual_sensitivity, left=np.add(total_sensitivity, interaction_sensitivity), color=self.individual_sensitivity_color,
                 label='Individual', height=self.default_height)
        self.configure_plot()
        if title: plt.title(title)
        plt.savefig(savePath) if savePath else plt.show()


if __name__ == "__main__":
    features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    individual_sensitivity = [0.1, 0.2, 0.15, 0.18]
    interaction_sensitivity = [0.05, 0.1, 0.08, 0.12]
    total_sensitivity = [0.15, 0.25, 0.2, 0.0]
    diag = GeneralizedTornadoDiagram(features)
    diag.plot_sensitivities(individual_sensitivity, interaction_sensitivity, total_sensitivity)
