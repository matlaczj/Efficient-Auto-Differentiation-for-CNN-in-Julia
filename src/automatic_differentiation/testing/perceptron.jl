include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
using LinearAlgebra

dense(w, b, x, activation) = activation(w * x .+ b)
dense(w, x, activation) = activation(w * x)
dense(w, x) = w * x

mean_squared_loss(y, ŷ) = Constant{Float64}(0.5) .* (y .- ŷ) .^ Constant{Float64}(2)

function net(x, wh, b, wo, y)
	x̂ = dense(wh, b, x, relu)
	x̂.name = "x̂"
	ŷ = dense(wo, x̂, relu)
	ŷ.name = "ŷ"
	E = mean_squared_loss(y, ŷ)
	E.name = "loss"

	return topological_sort(E)
end

Wh = Variable{Array{Float64, 2}}(randn(10, 2)::Array{Float64, 2}, name = "wh")
b = Variable{Vector{Float64}}(randn(10)::Vector{Float64}, name = "b")
Wo = Variable{Array{Float64, 2}}(randn(1, 10)::Array{Float64, 2}, name = "wo")
x = Variable{Vector{Float64}}([1.98, 4.434]::Vector{Float64}, name = "x")
y = Variable{Vector{Float64}}([0.064]::Vector{Float64}, name = "y")
losses = Float64[]

graph = net(x, Wh, b, Wo, y)

forward!(graph)
backward!(graph)
println("loss = $(graph[end].output)")

for (i, n) in enumerate(graph)
	if typeof(n) <: Variable
		println("Node $i")
		println(n.name)
		println(n.output)
		println(n.gradient)
		println()
	end
end

for (i, n) in enumerate(graph)
	print(i, ". ")
	println(n)
end
