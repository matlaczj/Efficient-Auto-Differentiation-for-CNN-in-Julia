include("basic_structures.jl")

# Updates the gradient of a constant node
function update_gradient!(node::Constant, gradient)
	return nothing
end

# Updates the gradient of a graph node. Accumulates gradients if the node has already been visited.
function update_gradient!(node::GraphNode, gradient)
	if isnothing(node.gradient)
		node.gradient = gradient
	else
		node.gradient .+= gradient
	end
end

# Computes the gradients of the nodes in reverse topological order using backpropagation.
function backward!(order::Vector; seed = 1.0)
	result = last(order)
	result.gradient = seed
	@assert length(result.output) == 1 "Gradient is defined only for scalar functions"
	for node in reverse(order)
		backward!(node)
	end
	return nothing
end

# Computes the gradient of a constant node (which is zero).
function backward!(node::Constant)
	return nothing
end

# Computes the gradient of a variable node (which is passed to it from the output node).
function backward!(node::Variable)
	return nothing
end

# Computes the gradient of an operator node, using the gradients of its output node(s) and the input node(s).
# Accumulates the input gradients using the update_gradient! function.
function backward!(node::Operator)
	inputs = node.inputs
	input_values = [input.output for input in inputs]
	gradient = node.gradient
	input_gradients = backward(node, input_values..., gradient)
	for (input, input_gradient) in zip(inputs, input_gradients)
		update_gradient!(input, input_gradient)
	end
	return nothing
end
