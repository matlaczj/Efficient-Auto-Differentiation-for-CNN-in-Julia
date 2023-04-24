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
	update_weights!(graph, learning_rate)
end

function build_graph()
	x = Variable(randn(8, 8, 1, 1), name = "x") # height, width, in_channels, batch_size
	wh = Variable(randn(3, 3, 1, 4), name = "wh") # # height, width, in_channels, out_channels
	bh = Variable(randn(1, 1, 4, 1), name = "bh")
	wo = Variable(randn(6 * 6 * 4, 1), name = "wo")
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

function train_model(x, y, learning_rate, n_iterations, if_print)
	losses = Vector{Float64}()
	graph, x_node, y_node = build_graph()

	# total number of iterations is n_iterations * batch_size
	for iter in 1:n_iterations
		if iter % 100 == 0
			println("Iteration $iter")
		# iterate over 4th dimention of x
		for i in 1:size(x, 4)
			# println("Sample $i")
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
			# get loss
			loss = graph[end].output[1]
			push!(losses, loss)
		end
	end
	return losses
end

n_samples = 100
height = 8
width = 8
channels = 1
x = randn(height, width, channels, n_samples)
y = randn(n_samples)
n_iterations = 1000
learning_rate = 0.1
losses = train_model(x, y, learning_rate, n_iterations, false)
println(losses)
# visualize loss
plot(losses, title = "Loss", xlabel = "Iteration", ylabel = "Loss")
