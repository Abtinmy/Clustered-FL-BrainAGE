import torch
from torch import nn


class Client:
    def __init__(self, index, model, lr, wd, device, train_dataloader, test_dataloader):
        self.index = index
        self.model = model
        self.lr = lr
        self.wd = wd
        self.device = device
        self.loss_func = nn.L1Loss(reduction='sum')
        self.train_dataloader = train_dataloader
        self.train_dataloader_iter = iter(train_dataloader)
        self.test_dataloader = test_dataloader
        self.test_dataloader_iter = iter(test_dataloader)

    def train_step(self):
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        batch_loss = []
        try:
            images, ages = next(self.train_dataloader_iter)
        except StopIteration:
            self.train_dataloader_iter = iter(self.train_dataloader)
            images, ages = next(self.train_dataloader_iter)

        images, ages = images.to(self.device).unsqueeze(1), ages.to(self.device).unsqueeze(1)
        self.model.zero_grad()
        pred_ages = self.model(images)
        loss = self.loss_func(pred_ages, ages)
        loss.backward()

        optimizer.step()
        batch_loss.append(loss.item())

        return sum(batch_loss), len(batch_loss)

    def inference(self, benchmark_dataloader):
        self.model.to(self.device)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for images, ages in benchmark_dataloader:
                images, ages = images.to(self.device).unsqueeze(1), ages.to(self.device).unsqueeze(1)
                pred_ages = self.model(images)
                predictions.extend(pred_ages.detach().cpu().numpy().tolist())

        return predictions

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_data_count(self, data_type):
        if data_type == 'train':
            return len(self.train_dataloader.dataset)
        elif data_type == 'test':
            return len(self.test_dataloader.dataset)
        else:
            print("invalid type.")
            exit(1)

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0
        with torch.no_grad():
            for images, ages in self.test_dataloader:
                images, ages = images.to(self.device).unsqueeze(1), ages.to(self.device).unsqueeze(1)
                pred_ages = self.model(images)
                test_loss += self.loss_func(pred_ages, ages).item()

        if len(self.test_dataloader.dataset) > 0:
            test_loss /= len(self.test_dataloader.dataset)
        else:
            test_loss = 0
        return test_loss
