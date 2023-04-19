include("basic_structures.jl")

# Topologically sorts a graph starting from a given node
function topological_sort(head::GraphNode)
	# Initialize sets to keep track of visited nodes and operators
	visited_nodes = Set{GraphNode}()
	visited_ops = Set{Operator}()

	# Initialize vector to keep the topologically sorted order
	order = Vector{GraphNode}()

	# Define a function to visit a node or operator
	visit(node::GraphNode) = (push!(visited_nodes, node); push!(order, node))
	visit(node::Operator) = begin
		if node ∉ visited_ops
			# Mark the operator as visited and recursively visit its inputs
			push!(visited_ops, node)
			for input in node.inputs
				visit(input)
			end
			# Push the operator to the topologically sorted order
			push!(order, node)
		end
	end

	# Start the topological sort from the given head node
	visit(head)

	# Return the topologically sorted order
	return order
end
