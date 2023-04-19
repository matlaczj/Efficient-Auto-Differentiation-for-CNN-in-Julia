include("basic_structures.jl")
import Base: ^, sin, sum, *, max
import LinearAlgebra: mul!

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
function forward(op::ScalarOperator{typeof(^)}, x, n)
	return x^n
end
function backward(op::ScalarOperator{typeof(^)}, x, n, gradient)
	term1 = gradient * n * x^(n - 1)
	term2 = gradient * log(abs(x)) * x^n
	return (term1, term2)
end

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
	let
		𝟏 = ones(length(node.output))
		Jx = diagm(y .* 𝟏)
		Jy = diagm(x .* 𝟏)
		tuple(Jx' * g, Jy' * g)
	end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
	let
		𝟏 = ones(length(x))
		J = 𝟏'
		tuple(J' * g)
	end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) =
	let
		𝟏 = ones(length(node.output))
		Jx = diagm(𝟏 ./ y)
		Jy = (-x ./ y .^ 2)
		tuple(Jx' * g, Jy' * g)
	end

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
	let
		Jx = diagm(isless.(y, x))
		Jy = diagm(isless.(x, y))
		tuple(Jx' * g, Jy' * g)
	end
