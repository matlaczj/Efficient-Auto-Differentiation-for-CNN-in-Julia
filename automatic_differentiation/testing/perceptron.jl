include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
using LinearAlgebra

Wh = Variable(randn(10, 2), name = "wh")
Wo = Variable(randn(1, 10), name = "wo")
x = Variable([1.98, 4.434], name = "x")
y = Variable([0.064], name = "y")
losses = Float64[]

function dense(w, b, x, activation)
	return activation(w * x .+ b)
end
function dense(w, x, activation)
	return activation(w * x)
end
function dense(w, x)
	return w * x
end

function mean_squared_loss(y, ŷ)
	return Constant(0.5) .* (y .- ŷ) .^ Constant(2)
end

function net(x, wh, wo, y)
	x̂ = dense(wh, x, relu)
	x̂.name = "x̂"
	ŷ = dense(wo, x̂, relu)
	ŷ.name = "ŷ"
	E = mean_squared_loss(y, ŷ)
	E.name = "loss"

	return topological_sort(E)
end
graph = net(x, Wh, Wo, y)
forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
	print(i, ". ")
	println(n)
end