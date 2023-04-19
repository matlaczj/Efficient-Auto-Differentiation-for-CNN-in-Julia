import Base: show, summary

abstract type GraphNode end

abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
	output::T
end

mutable struct Variable <: GraphNode
	output::Any
	gradient::Any
	name::String
	function Variable(output; name = "?")
		new(output, nothing, name)
	end
end

mutable struct ScalarOperator{F} <: Operator
	inputs::Any
	output::Any
	gradient::Any
	name::String
	function ScalarOperator(fun, inputs...; name = "?")
		new{typeof(fun)}(inputs, nothing, nothing, name)
	end
end

mutable struct BroadcastedOperator{F} <: Operator
	inputs::Any
	output::Any
	gradient::Any
	name::String
	function BroadcastedOperator(fun, inputs...; name = "?")
		new{typeof(fun)}(inputs, nothing, nothing, name)
	end
end

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
	print(io, "var ", x.name)
	print(io, "\n ┣━ ^ ")
	summary(io, x.output)
	print(io, "\n ┗━ ∇ ")
	summary(io, x.gradient)
end

