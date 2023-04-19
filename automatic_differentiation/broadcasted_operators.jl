include("basic_structures.jl")
import Base: ^, sin, sum, *, +, -, max
import LinearAlgebra: mul!

^(x::GraphNode, n::Number) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(g .* n .* x .^ (n - 1), nothing)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

sigmoid(x::GraphNode) = BroadcastedOperator(sigmoid, x)
forward(::BroadcastedOperator{typeof(sigmoid)}, x) = 1 ./ (1 .+ exp.(-x))
backward(::BroadcastedOperator{typeof(sigmoid)}, x, g) = tuple(g .* sigmoid(x) .* (1 .- sigmoid(x)))

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, 0)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* isless.(x, 0))

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
	let
		𝟏 = ones(length(node.output))
		Jx = diagm(y .* 𝟏)
		Jy = diagm(x .* 𝟏)
		tuple(Jx' * g, Jy' * g)
	end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
	let
		𝟏 = ones(length(x))
		J = 𝟏'
		tuple(J' * g)
	end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) =
	let
		𝟏 = ones(length(node.output))
		Jx = diagm(𝟏 ./ y)
		Jy = (-x ./ y .^ 2)
		tuple(Jx' * g, Jy' * g)
	end

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
	let
		Jx = diagm(isless.(y, x))
		Jy = diagm(isless.(x, y))
		tuple(Jx' * g, Jy' * g)
	end

