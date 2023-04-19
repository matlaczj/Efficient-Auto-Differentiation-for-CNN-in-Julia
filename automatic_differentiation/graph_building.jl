include("basic_structures.jl")

function topological_sort(head::GraphNode)
	visited_nodes = Set{GraphNode}()
	visited_ops = Set{Operator}()
	order = Vector{GraphNode}()
	visit(node::GraphNode) = (push!(visited_nodes, node); push!(order, node))
	visit(node::Operator) = begin
		if node ∉ visited_ops
			push!(visited_ops, node)
			for input in node.inputs
				visit(input)
			end
			push!(order, node)
		end
	end
	visit(head)
	return order
end
