using MLDatasets

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Filter out images that are not of the digit 5
train_mask = train_y .== 5
test_mask = test_y .== 5
train_x = train_x[:, :, train_mask]
test_x = test_x[:, :, test_mask]

using Images

# Rescale images to 8x8
train_x = map(img -> imresize(img, (8, 8)), train_x)
test_x = map(img -> imresize(img, (8, 8)), test_x)

train_x = reshape(train_x, (8, 8, 1, :))
test_x = reshape(test_x, (8, 8, 1, :))

# Stack images into batches with batch_size 1
train_x = cat(train_x..., dims = 4)
test_x = cat(test_x..., dims = 4)

train_x = Gray.(train_x)
test_x = Gray.(test_x)

train_y = train_y[train_mask] .== 5
test_y = test_y[test_mask] .== 5

using Random

# Find the class with fewer samples
minority_class = findfirst(train_y .== 1)
majority_class = findfirst(train_y .== 0)

# Undersample the majority class
n_minority = sum(train_y)
n_majority = length(train_y) - n_minority
rand_idxs = randperm(n_majority)[1:n_minority]
train_x = cat(train_x[:, :, minority_class, :], train_x[:, :, majority_class, rand_idxs], dims = 4)
train_y = vcat(fill(true, n_minority), fill(false, n_minority))
