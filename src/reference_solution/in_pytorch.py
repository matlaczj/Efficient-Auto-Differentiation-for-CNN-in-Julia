import torch
import torch.nn as nn
from algorytmy.reference_solution.config import *
from algorytmy.reference_solution.mnist_dataset import train_images, train_labels, test_images, test_labels
import time

message = "Running on GPU" if torch.cuda.is_available() else "Running on CPU"
print(message)


class Model(nn.Module):
    def __init__(
        self,
        num_filters_1,
        kernel_size,
        input_shape,
        num_units,
        activation_1,
    ):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[-1],
            out_channels=num_filters_1,
            kernel_size=kernel_size,
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            # NOTE: Because pytorch can't handle this dynamically... This is bad.
            in_features=2704,
            out_features=num_units,
        )
        self.activation_1 = nn.ReLU() if activation_1 == "relu" else None

    def forward(self, x):
        # NOTE: PyTorch expects the input to be in the format (batch_size, channels, height, width)
        # while TensorFlow expects the input to be in the format (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.activation_1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation_1(x)
        return x

    def summary(self):
        print(self)


model = Model(
    num_filters_1,
    kernel_size,
    input_shape,
    num_units,
    activation_1,
)
model.summary()

optimizer = (
    torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer == "sgd"
    else None
)
loss_fn = nn.MSELoss() if loss == "mean_squared_error" else None

train_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(train_images), torch.LongTensor(train_labels)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(test_images), torch.LongTensor(test_labels)
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

start_time = time.time()
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Batch {}/{}".format(batch_idx + 1, len(train_loader)), end=", ")
        optimizer.zero_grad()
        output = model(data)
        output = torch.max(output, 1)[0]
        loss = loss_fn(output.float(), target.float())
        loss.backward()
        optimizer.step()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

model.eval()