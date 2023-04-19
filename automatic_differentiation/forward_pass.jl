# Include basic structures file
include("basic_structures.jl")

# Define a function to reset the gradient of a constant node (does nothing)
function reset!(node::Constant)
	return nothing
end

# Define a function to reset the gradient of a variable node
function reset!(node::Variable)
	return node.gradient = nothing
end

# Define a function to reset the gradient of an operator node
function reset!(node::Operator)
	return node.gradient = nothing
end

# Define a function to compute the output of a constant node
function compute!(node::Constant)
	return node.output
end

# Define a function to compute the output of a variable node
function compute!(node::Variable)
	return node.output
end

# Define a function to compute the output of an operator node
function compute!(node::Operator)
	# Get the inputs to the operator node
	inputs = [input.output for input in node.inputs]
	# Compute the output of the operator node using the forward function
	node.output = forward(node, inputs...)
	# Return the output of the operator node
	return node.output
end

# Define a function to perform forward propagation on a vector of nodes
function forward!(order::Vector{<:GraphNode})
	# Iterate over the nodes in the order provided
	for node in order
		# Compute the output of the current node
		compute!(node)
		# Reset the gradient of the current node
		reset!(node)
	end
	# Return the output of the last node in the order
	return last(order).output
end
