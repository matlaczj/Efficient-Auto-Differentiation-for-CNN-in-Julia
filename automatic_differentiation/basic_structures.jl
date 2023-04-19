# Define an abstract type for all graph nodes
abstract type GraphNode end

# Define an abstract type for all operators (subtypes of graph nodes)
abstract type Operator <: GraphNode end

# Define a type for constant values
struct Constant{T} <: GraphNode
	output::T
end

# Define a mutable type for variables
mutable struct Variable <: GraphNode
	output::Any
	gradient::Any
	name::String
	# Constructor for creating a new variable
	function Variable(output; name = "?")
		new(output, nothing, name)
	end
end

# Define a mutable type for scalar operators
mutable struct ScalarOperator{F} <: Operator
	inputs::Any
	output::Any
	gradient::Any
	name::String
	# Constructor for creating a new scalar operator
	function ScalarOperator(fun, inputs...; name = "?")
		new{typeof(fun)}(inputs, nothing, nothing, name)
	end
end

# Define a mutable type for broadcasted operators
mutable struct BroadcastedOperator{F} <: Operator
	inputs::Any
	output::Any
	gradient::Any
	name::String
	# Constructor for creating a new broadcasted operator
	function BroadcastedOperator(fun, inputs...; name = "?")
		new{typeof(fun)}(inputs, nothing, nothing, name)
	end
end

# Define how to display a scalar operator
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");

# Define how to display a broadcasted operator
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");

# Define how to display a constant value
show(io::IO, x::Constant) = print(io, "const ", x.output)

# Define how to display a variable, including its gradient
show(io::IO, x::Variable) = begin
	print(io, "var ", x.name)
	print(io, "\n ┣━ ^ ")
	summary(io, x.output)
	print(io, "\n ┗━ ∇ ")
	summary(io, x.gradient)
end
