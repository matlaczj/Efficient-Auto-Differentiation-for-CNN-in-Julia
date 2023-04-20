include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
include("../convolution.jl")
using LinearAlgebra

function conv(w, b, x, activation)
	out = conv(x, w) .+ b
	return activation(out)
end
function dense(w, b, x, activation)
	return activation(w * x .+ b)
end
function dense(w, x, activation)
	return activation(w * x)
end
function dense(w, x)
	# NOTE: Other than in MLP example but otherwise we get huge output in the last layer.
	return x * w
end
function mean_squared_loss(y, ŷ)
	return Constant(0.5) .* (y .- ŷ) .^ Constant(2)
end
function flatten(x)
	return flatten(x)
end

function net(x, wh, bh, wo, y)
	x̂ = conv(wh, bh, x, relu) # out_channels, height_out, width_out, batch_size
	x̂.name = "x̂"
	x̂ = flatten(x̂)
	x̂.name = "x̂"
	ŷ = dense(wo, x̂)
	ŷ.name = "ŷ"
	e = mean_squared_loss(y, ŷ)
	e.name = "loss"
	return topological_sort(e)
end

x = Variable(randn(8, 8, 1, 1), name = "x") # height, width, in_channels, batch_size
wh = Variable(randn(3, 3, 1, 4), name = "wh") # height, width, in_channels, out_channels
bh = Variable(randn(6), name = "bh")
wo = Variable(randn(6 * 6 * 4, 1), name = "wo")
y = Variable(randn(1), name = "y")

graph = net(x, wh, bh, wo, y)

for (i, n) in enumerate(graph)
	if typeof(n) <: Variable
		println("Node $i")
		println(n.name)
		println(n.output)
		println(n.gradient)
		println()
	end
end

println("Forward pass")

for (i, n) in enumerate(graph)
	print(i, ". ")
	println(n)
end


forward!(graph)
backward!(graph)
