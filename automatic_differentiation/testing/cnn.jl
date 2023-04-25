include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
include("../convolution.jl")
using LinearAlgebra
using Statistics
using Plots
using MLDatasets: MNIST
using Images

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end
function dense(w, b, x, activation)
	return activation((x * w) .+ b)
end
function mean_squared_loss(y, ŷ)
	return Constant(0.5) .* (y .- ŷ) .^ Constant(2)
end
function flatten(x)
	return flatten(x)
end

function update_weight!(node, learning_rate)
	node.output -= learning_rate .* node.gradient
end

function is_weight(node)
	return occursin("w", node.name)
end

function is_bias(node)
	return occursin("b", node.name)
end

function is_parameter(node)
	return is_weight(node) || is_bias(node)
end

function is_x(node)
	return occursin("x", node.name)
end

function is_y(node)
	return occursin("y", node.name)
end

function has_name(node)
	return hasproperty(node, :name)
end

function average_bias_gradient!(node)
	node.gradient = mean(node.gradient, dims = (1, 2))
end

function update_weights!(graph, learning_rate)
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			update_weight!(node, learning_rate)
		end
	end
end

function save_param_gradients!(graph, gradients_in_batch)
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			if !haskey(gradients_in_batch, node.name)
				gradients_in_batch[node.name] = Vector{Any}()
			end
			push!(gradients_in_batch[node.name], node.gradient)
		end
	end
end

function learning_iteration!(graph, learning_rate, if_print)
	forward!(graph)
	backward!(graph)
	if if_print
		for (i, n) in enumerate(graph)
			if typeof(n) <: Variable
				println("Node $i")
				println(n.name)
				println(n.output)
				println(size(n.output))
				println(n.gradient)
				println()
			end
		end
	end
	# update_weights!(graph, learning_rate)
end

function build_graph()
	input_size = 28
	kernel_size = 3
	input_channels = 1
	out_channels = 4
	x = Variable(randn(input_size, input_size, input_channels, 1), name = "x") # height, width, in_channels, batch_size
	wh = Variable(randn(kernel_size, kernel_size, input_channels, out_channels), name = "wh") # # height, width, in_channels, out_channels
	bh = Variable(randn(1, 1, out_channels, 1), name = "bh")
	wo = Variable(randn((input_size-2) * (input_size-2) * out_channels, 1), name = "wo")
	bo = Variable(randn(1, 1), name = "bo")
	y = Variable(randn(1), name = "y")

	x̂ = conv(wh, bh, x, relu)
	x̂.name = "x̂"
	x̂ = flatten(x̂)
	x̂.name = "x̂"
	ŷ = dense(wo, bo, x̂, logistic)
	ŷ.name = "ŷ"
	e = mean_squared_loss(y, ŷ)
	e.name = "loss"
	return topological_sort(e), x, y
end

# calculate mean of list of same size and shape matrixes
# by adding them together and dividing by number of matrixes
function mean_of_matrixes(matrixes::Vector{Any})
	sum = matrixes[1]
	for i in 2:length(matrixes)
		sum .+= matrixes[i]
	end
	return sum ./ length(matrixes)
end

function train_model(x, y, learning_rate, n_iterations, if_print)
	graph, x_node, y_node = build_graph()
	avg_losses = Vector{Float64}()
	
	# total number of iterations is n_iterations * batch_size
	for iter in 1:n_iterations
		losses_in_batch = Vector{Float64}()
		gradients_in_batch = Dict{String, Vector{Any}}()
		mean_gradients = Dict{String, Any}()
		println("Iteration $iter")
		# iterate over 4th dimention of x
		for i in 1:size(x, 4)
			println("Sample $i")
			# current x as a 4d tensor
			curr_x = x[:, :, :, i]
			curr_x = reshape(curr_x, size(curr_x, 1), size(curr_x, 2), size(curr_x, 3), 1)
			# current y as a vector
			curr_y = y[i, :]
			# replace x and y nodes output
			x_node.output = curr_x
			y_node.output = curr_y
			# run learning iteration
			learning_iteration!(graph, learning_rate, if_print)
			# save parameter gradients
			save_param_gradients!(graph, gradients_in_batch)
			# get loss
			loss = graph[end].output[1]
			push!(losses_in_batch, loss)
		end
		# print gradients_in_batch
		for (key, value) in gradients_in_batch
			println(key)
			println(size(gradients_in_batch[key]))
			for i in 1:length(gradients_in_batch[key])
				println(size(gradients_in_batch[key][i]))
			end
		end
		# update parameters
		for (key, value) in gradients_in_batch
			println("update parameters--->", key)
			# println(size(gradients_in_batch[key]))
			println("gradients_in_batch[key] --->", typeof(gradients_in_batch[key]))
			matrixes = gradients_in_batch[key]
			sum = matrixes[1]
			for i in 2:length(matrixes)
				sum .+= matrixes[i]
			end
			mean = sum ./ length(matrixes)
			println(size(mean))
			println("mean --->", typeof(mean))
			mean_gradients[key] = mean
			# println(size(gradients_in_batch[key]))
		end
		for (idx, node) in enumerate(graph)
			if has_name(node) && is_parameter(node)
				println("learning_rate--->", node.name)
				print(size(node.output))
				print(size(mean_gradients[node.name]))
				node.output -= learning_rate .* mean_gradients[node.name]
			end
		end
		# get average loss
		avg_loss = mean(losses_in_batch)
		push!(avg_losses, avg_loss)
	end
	return avg_losses
end

train_ds = MNIST(:train)
x_train = train_ds.features
y_train = train_ds.targets
x_train = reshape(x_train, size(x_train, 1), size(x_train, 2), 1, size(x_train, 3))

# modify target 'y_train' to be binary classification of 5 or not 5
y_train = y_train .== 5

# take only 100 first samples of train datasets
x_train = x_train[:, :, :, 1:10]
y_train = y_train[1:10, :]
x = x_train
y = y_train

# n_samples = 2
# height = 8
# width = 8
# channels = 1
# x = randn(height, width, channels, n_samples)
# y = randn(n_samples)
n_iterations = 100
learning_rate = 0.1
losses = train_model(x, y, learning_rate, n_iterations, false)
# visualize loss
plot(losses, title = "Loss", xlabel = "Iteration", ylabel = "Loss")