import matplotlib.pyplot as plt


# todo: plot axvline at the output of base case prediction

class SiderPlots(object):
    def __init__(self, variables_names, preview=True, save_fig=False, filename=None):
        self.preview = preview
        self.save_fig = save_fig
        self.filename = filename
        self.variables_names = variables_names

        self.default_figsize = (8, 4)
        self.default_x_lablel = "Predetermined Range"
        self.default_title = "Sensitivity Analysis of IoI"
        self.default_y_lablel = "One Way Sensitivity Function"

    def plot_predetermined_predictions(self, predetermined_predictions, title: str = "", savePath: str=""):
        x = predetermined_predictions['x']
        y = predetermined_predictions['y']
        plt.figure(figsize=self.default_figsize)
        for i in range(len(self.variables_names)):
            plt.plot(x[i, :, i], y[i, :], label=f"{self.variables_names[i]}")
        self.configure_plot()
        if title: plt.title(title)
        plt.savefig(savePath) if savePath else plt.show()

    def configure_plot(self):
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel(self.default_x_lablel)
        plt.ylabel(self.default_y_lablel)
        plt.title(self.default_title)
        plt.legend(loc='upper right')  # Show legend with labels
        plt.tight_layout()
        plt.grid(True)