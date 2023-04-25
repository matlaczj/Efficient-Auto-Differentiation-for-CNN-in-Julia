using MLDatasets: MNIST
train_ds = MNIST(:train)
test_ds = MNIST(:test)
x_train = train_ds.features
y_train = train_ds.targets
x_test = test_ds.features
y_test = test_ds.targets