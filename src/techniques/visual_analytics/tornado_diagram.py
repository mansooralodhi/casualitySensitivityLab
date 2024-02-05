import matplotlib.pyplot as plt
from typing import List

# todo: plot bi-direction tonrado diagram to plot lower and upper bound in same digram

class TornadoDiagram(object):
    def __init__(self, variables_names, preview=True, save_fig=False, filename=None):
        self.preview = preview
        self.save_fig = save_fig
        self.filename = filename
        self.variables_names = variables_names

        self.default_figsize = (8,4)
        self.default_color = "skyblue"
        self.default_x_lablel = "Features Impact"
        self.default_title="Sensitivity Analysis of IoI"

    def configure_plot(self):
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel(self.default_x_lablel)
        plt.title("Lower Bond Sensitivity Measure")
        plt.legend(loc='upper right')
        plt.tight_layout()

    def plot_sensitivity(self, impacts: list, title: str = "", savePath: str =""):
        if len(impacts) != len(self.variables_names):
            raise Exception("Number of Impacts is Not Equal to Number of Variables !")
        plt.figure(figsize=self.default_figsize)
        plt.barh(self.variables_names, impacts, color=self.default_color, alpha=0.7, label='Influence')
        self.configure_plot()
        if title: plt.title(title)
        plt.savefig(savePath) if savePath else plt.show()

    def plot_boundary_sensitivities(self, lower_boundary_impact: List[float], upper_boundary_impact: List[float],
                                    title: str = "", savePath: str =""):
        if len(lower_boundary_impact) != len(upper_boundary_impact) != len(self.variables_names):
            raise Exception("Number of Impacts is Not Equal to Number of Variables !")
        plt.figure(figsize=self.default_figsize)
        plt.barh(self.variables_names, lower_boundary_impact, color='red', alpha=0.7, label='Lower Bound')
        plt.barh(self.variables_names, upper_boundary_impact, left=lower_boundary_impact, color='blue', alpha=0.7, label='Upper Bound')
        self.configure_plot()
        if title: plt.title(title)
        plt.savefig(savePath) if savePath else plt.show()

