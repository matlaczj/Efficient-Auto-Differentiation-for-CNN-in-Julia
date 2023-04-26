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
using Random
Random.seed!(1)

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end
dense(w, b, x, activation) = activation((x * w) .+ b)
mean_squared_loss(y, ŷ) = Constant(0.5) .* (y .- ŷ) .^ Constant(2)
flatten(x) = flatten(x)

update_weight!(node, learning_rate) = node.output -= learning_rate .* node.gradient

is_weight(node) = occursin("w", node.name)

is_bias(node) = occursin("b", node.name)

is_parameter(node) = is_weight(node) || is_bias(node)

is_x(node) = occursin("x", node.name)

is_y(node) = occursin("y", node.name)

has_name(node) = hasproperty(node, :name)

average_bias_gradient!(node) = node.gradient = mean(node.gradient, dims = (1, 2))

update_weights!(graph, learning_rate) =
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			update_weight!(node, learning_rate)
		end
	end

function save_param_gradients!(graph, gradients_in_batch)
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			if !haskey(gradients_in_batch, node.name)
				gradients_in_batch[node.name] = Vector{AbstractArray{<:Real, 4}}()
			end
			push!(gradients_in_batch[node.name], node.gradient)
			# println("save_param_gradients!: ", typeof(node.gradient))
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
end

function build_graph()
	input_size = 28
	kernel_size = 3
	input_channels = 1
	out_channels = 4
	x = Variable(randn(input_size, input_size, input_channels, 1), name = "x")
	wh =
		Variable(randn(kernel_size, kernel_size, input_channels, out_channels), name = "wh")
	bh = Variable(randn(1, 1, out_channels, 1), name = "bh")
	wo = Variable(randn((input_size - 2) * (input_size - 2) * out_channels, 1), name = "wo")
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

# Define a function to train a neural network model
function train_model(x, y, learning_rate, n_iterations, if_print)
	# Build the computation graph and initialize the input nodes
	graph, x_node, y_node = build_graph()
	# Create a vector to store the average loss after each iteration
	avg_losses = Vector{Float64}()

	# Iterate over the specified number of batches
	for iter ∈ 1:n_iterations
		# Create empty vectors to store the losses and gradients in the current batch
		losses_in_batch = Vector{Float64}()
		gradients_in_batch = Dict{String, Vector{AbstractArray}}()
		# Create a dictionary to store the mean gradients across the batch
		mean_gradients = Dict{String, AbstractArray}()
		# Print the batch number
		println("Batch $iter")

		# Iterate over each input in the batch
		for i ∈ 1:size(x, 4)
			# Get the current input and reshape it to match the input node shape
			curr_x = x[:, :, :, i]
			curr_x = reshape(curr_x, size(curr_x, 1), size(curr_x, 2), size(curr_x, 3), 1)
			# Get the current target output
			curr_y = y[i, :]
			# Set the input and output nodes to the current values
			x_node.output = curr_x
			y_node.output = curr_y
			# Perform a forward pass through the graph and update the gradients
			learning_iteration!(graph, learning_rate, if_print)
			# Save the gradients for each parameter in the graph
			save_param_gradients!(graph, gradients_in_batch)
			# Get the loss for the current input and append it to the losses in the batch vector
			loss = graph[end].output[1]
			push!(losses_in_batch, loss)
		end

		# Compute the mean gradients for each parameter in the graph
		for (key, value) in gradients_in_batch
			matrixes = gradients_in_batch[key]
			sum = matrixes[1]
			n_matrixes = length(matrixes)
			for i ∈ 2:n_matrixes
				sum .+= matrixes[i]
			end
			mean = sum ./ n_matrixes
			mean_gradients[key] = mean
		end

		# Update each parameter in the graph using the mean gradients
		for (idx, node) in enumerate(graph)
			if has_name(node) && is_parameter(node)
				node.output -= learning_rate .* mean_gradients[node.name]
			end
		end

		# Compute the average loss for the batch and append it to the average losses vector
		avg_loss = mean(losses_in_batch)
		push!(avg_losses, avg_loss)
	end
	# Return the vector of average losses
	return avg_losses, graph
end

train_ds = MNIST(:train)
x_train = train_ds.features
y_train = train_ds.targets
x_train = reshape(x_train, size(x_train, 1), size(x_train, 2), 1, size(x_train, 3))
y_train = y_train .== 5
x_train = x_train[:, :, :, 1:100]
y_train = y_train[1:100, :]
x = x_train
y = y_train

n_batches = 100
learning_rate = 0.1
start = time()
losses, graph = train_model(x, y, learning_rate, n_batches, false)
finish = time()
println("Time of execution: ", finish - start)
plot(losses, title = "Loss", xlabel = "Iteration", ylabel = "Loss")