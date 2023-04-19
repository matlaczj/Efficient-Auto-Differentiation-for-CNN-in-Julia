include("basic_structures.jl")

update_gradient!(node::Constant, gradient) = nothing

update_gradient!(node::GraphNode, gradient) = begin
	if isnothing(node.gradient)
		node.gradient = gradient
	else
		node.gradient .+= gradient
	end
end

function backward!(order::Vector; seed = 1.0)
	result = last(order)
	result.gradient = seed
	@assert length(result.output) == 1 "Gradient is defined only for scalar functions"
	for node in reverse(order)
		backward!(node)
	end
	return nothing
end

function backward!(node::Constant)
	return nothing
end

function backward!(node::Variable)
	return nothing
end

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
