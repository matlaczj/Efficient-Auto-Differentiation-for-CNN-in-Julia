include("basic_structures.jl")

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

function compute!(node::Constant)
	return node.output
end

function compute!(node::Variable)
	return node.output
end

function compute!(node::Operator)
	inputs = [input.output for input in node.inputs]
	node.output = forward(node, inputs...)
	return node.output
end

function forward!(order::Vector{<:GraphNode})
	for node in order
		compute!(node)
		reset!(node)
	end
	return last(order).output
end
