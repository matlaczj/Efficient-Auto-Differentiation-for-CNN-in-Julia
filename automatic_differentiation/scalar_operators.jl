include("basic_structures.jl")
import Base: ^, *, +, -, sin, max, min

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = x^n
backward(::ScalarOperator{typeof(^)}, x, n, gradient) = begin
	term1 = gradient * n * x^(n - 1)
	term2 = gradient * log(abs(x)) * x^n
	return (term1, term2)
end

+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
forward(::ScalarOperator{typeof(+)}, x, y) = x + y
backward(::ScalarOperator{typeof(+)}, x, y, g) = tuple(g, g)

-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
forward(::ScalarOperator{typeof(-)}, x, y) = x - y
backward(::ScalarOperator{typeof(-)}, x, y, g) = tuple(g, -g)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
forward(::ScalarOperator{typeof(*)}, x, y) = x * y
backward(::ScalarOperator{typeof(*)}, x, y, g) = tuple(y' * g, x' * g)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

max(x::GraphNode, y::GraphNode) = ScalarOperator(max, x, y)
forward(::ScalarOperator{typeof(max)}, x, y) = max(x, y)
backward(::ScalarOperator{typeof(max)}, x, y, g) = begin
	term1 = g * isless(y, x)
	term2 = g * isless(x, y)
	return (term1, term2)
end

min(x::GraphNode, y::GraphNode) = ScalarOperator(min, x, y)
forward(::ScalarOperator{typeof(min)}, x, y) = min(x, y)
backward(::ScalarOperator{typeof(min)}, x, y, g) = begin
	term1 = g * isless(x, y)
	term2 = g * isless(y, x)
	return (term1, term2)
end

sigmoid(x::GraphNode) = ScalarOperator(sigmoid, x)
forward(::ScalarOperator{typeof(sigmoid)}, x) = 1 / (1 + exp(-x))
backward(::ScalarOperator{typeof(sigmoid)}, x, g) = g * sigmoid(x) * (1 - sigmoid(x))
