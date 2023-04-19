include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")

x = Variable(5.0, name = "x")
two = Constant(2.0)
squared = x^two
sine = sin(squared)

order = topological_sort(sine)

y = forward!(order)

if y == -0.13235175009777303
	println("Test passed")
else
	println("Test failed")
end
println("y = $y")


backward!(order)

if x.gradient == 9.912028118634735
	println("Test passed")
else
	println("Test failed")
end
println("∂y/∂x = $(x.gradient)")
