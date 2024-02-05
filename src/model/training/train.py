import torch.optim
import torch.nn as nn

from utils import *
from src.data import *
from src.data.dataset import Dataset
from src.model.architectures import *


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.dataset = eval(self.args.get("dataset_name"))(scale_data=True, split_data=True)
        self.loss_function = nn.L1Loss()
        self.model = eval(self.args.get("model_name"))(self.dataset.X.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.get("lr"))

        self.trainDataloader = None
        self.testDataloader = None
        self.construct_dataloader()

    def construct_dataloader(self):
        train_dataset = Dataset(self.dataset.X_train, self.dataset.y_train)
        test_dataset = Dataset(self.dataset.X_test, self.dataset.y_test)
        self.trainDataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.get("batch_size"),
                                                           shuffle=True, num_workers=self.args.get("num_workers"))
        self.testDataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.get("batch_size"),
                                                          shuffle=True, num_workers=self.args.get("num_workers"))

    def train(self):

        full_train_loss, full_test_loss = list(), list()
        n = 1

        for _ in range(self.args.get("epochs")):
            train_loss, test_loss = list(), list()

            for i, data in enumerate(self.trainDataloader, 1):
                inputs, targets = data
                targets = targets.reshape((targets.shape[0], 1))
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                if i % 1 == 0:
                    full_train_loss.append(sum(train_loss) / len(train_loss))
                    self.model.eval()
                    with torch.no_grad():
                        for j, data in enumerate(self.testDataloader, 1):
                            inputs, targets = data
                            targets = targets.reshape((targets.shape[0], 1))
                            outputs = self.model(inputs)
                            loss = self.loss_function(outputs, targets)
                            test_loss.append(loss.item())
                        full_test_loss.append(sum(test_loss) / len(test_loss))

                    print(f"Step {n} :  Train Loss {sum(train_loss) / len(train_loss)} ; "
                          f"Test Loss {sum(test_loss) / len(test_loss)}")
                    train_loss, test_loss, n = list(), list(), n + 1

        plot_loss(full_train_loss, full_test_loss)
        torch.save(self.model.state_dict(), self.args.get("model_filename"))


if __name__ == '__main__':
    parameters = dict(loss_filename="../artifacts/non_linear_regressor_loss.png",
                      model_filename="../artifacts/non_linear_regression_model.pth",
                      dataset_name="CaliforniaHousingDataset",
                      model_name="NonlinearRegressionModel",
                      batch_size=100,
                      epochs=1,
                      num_workers=1,
                      lr=0.0004, )

    Trainer(parameters).train()
