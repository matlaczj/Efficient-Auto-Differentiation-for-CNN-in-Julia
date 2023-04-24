include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
include("../convolution.jl")
using LinearAlgebra
using Statistics

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

function net(x, wh, bh, wo, bo, y)
	x̂ = conv(wh, bh, x, relu) # out_channels, height_out, width_out, batch_size
	x̂.name = "x̂"
	x̂ = flatten(x̂)
	x̂.name = "x̂"
	ŷ = dense(wo, bo, x̂, logistic) # add activation like softmax
	ŷ.name = "ŷ"
	e = mean_squared_loss(y, ŷ)
	e.name = "loss"
	return topological_sort(e)
end



# for (i, n) in enumerate(graph)
# 	if typeof(n) <: Variable
# 		println("Node $i")
# 		println(n.name)
# 		println(n.output)
# 		println(n.gradient)
# 		println()
# 	end
# end

# println("Forward pass")






# function that adds gradient to the node's weight
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

function has_name(node)
	return hasproperty(node, :name)
end

# average out the gradient for biases to reduce its dimensionality
function average_bias_gradient!(node)
	node.gradient = mean(node.gradient, dims = (1, 2))
end

# run update on all weights and biases in the graph
# except for the input node and the loss node
function update_weights!(graph, learning_rate)
	for (idx, node) in enumerate(graph)
		if has_name(node) && is_parameter(node)
			if is_bias(node)
				average_bias_gradient!(node)
			end
			println(idx)
			println(node.name)
			update_weight!(node, learning_rate)
		end
	end
end

x = Variable(randn(8, 8, 1, 1), name = "x") # height, width, in_channels, batch_size
wh = Variable(randn(3, 3, 1, 4), name = "wh") # height, width, in_channels, out_channels
bh = Variable(randn(1, 1, 4, 1), name = "bh")
wo = Variable(randn(6 * 6 * 4, 1), name = "wo")
bo = Variable(randn(1, 1), name = "bo")
y = Variable(randn(1), name = "y")

graph = net(x, wh, bh, wo, bo, y)

forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
	if typeof(n) <: Variable
		println("Node $i")
		println(n.name)
		println(n.output)
		println(size(n.output))
		println(n.gradient)
		println()
	end
	# print(i, ". ")
	# println(n)
end

update_weights!(graph, 0.1)




graph[6]

graph[6].output

graph[6].gradient

graph[4]

graph[4].output

graph[4].gradient

graph[5]

graph[5].output

graph[5].gradient

graph[6].output # add bias to each in_channel

graph[3].output .+ randn(1)

average_bias_gradient!(graph[6])