import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss):
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.savefig("..//artifacts///train_loss.png")
    plt.show()
