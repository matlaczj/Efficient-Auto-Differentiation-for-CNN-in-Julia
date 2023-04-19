include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
using LinearAlgebra

Wh = Variable(randn(10, 2), name = "wh")
Wo = Variable(randn(1, 10), name = "wo")
x = Variable([1.98, 4.434], name = "x")
y = Variable([0.064], name = "y")
losses = Float64[]

function activation(x)
	return 1 ./ (1 .+ exp.(-x))
end
function dense(w, b, x, activation)
	return activation(w * x .+ b)
end
function dense(w, x, activation)
	return activation(w * x)
end
function dense(w, x)
	return w * x
end

function mean_squared_loss(y, ŷ)
	return Constant(0.5) .* (y .- ŷ) .^ Constant(2)
end

function net(x, wh, wo, y)
	x̂ = dense(wh, x)
	x̂.name = "x̂"
	ŷ = dense(wo, x̂)
	ŷ.name = "ŷ"
	E = mean_squared_loss(y, ŷ)
	E.name = "loss"

	return topological_sort(E)
end
graph = net(x, Wh, Wo, y)
forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
	print(i, ". ")
	println(n)
end

function conv2d(w, x; stride = (1, 1), padding = (0, 0))
	# w: (out_channels, in_channels, kernel_height, kernel_width)
	# x: (batch_size, in_channels, height, width)

	# Get shapes
	batch_size, in_channels, height, width = size(x)
	out_channels, _, kernel_height, kernel_width = size(w)

	# Add padding to input data
	x = padarray(x, padding)

	# Reshape input data
	x = reshape(x, (batch_size * height * width, in_channels))

	# Reshape weights
	w = reshape(w, (out_channels, in_channels * kernel_height * kernel_width))

	# Perform matrix multiplication
	y = w * x'

	# Reshape output
	y = reshape(y, (out_channels, batch_size, height, width))
	y = permutedims(y, (2, 1, 3, 4))

	# Apply stride
	y = y[:, :, 1:stride[1]:end, 1:stride[2]:end]

	return y
end
