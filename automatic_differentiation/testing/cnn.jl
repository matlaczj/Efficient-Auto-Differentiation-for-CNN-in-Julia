include("../basic_structures.jl")
include("../graph_building.jl")
include("../forward_pass.jl")
include("../backward_pass.jl")
include("../scalar_operators.jl")
include("../broadcasted_operators.jl")
using LinearAlgebra

@inline function im2col(A, n, m)
	M, N = size(A)
	B = Array{eltype(A)}(undef, m * n,
		(M - m + 1) * (N - n + 1))
	indx = reshape(1:M*N, M, N)[1:M-m+1, 1:N-n+1]
	for (i, value) in enumerate(indx)
		for j ∈ 0:n-1
			@views B[(i-1)*m*n+j*m+1:(i-1)m*n+(j+1)m] =
				A[value+j*M:value+m-1+j*M]
		end
	end
	return B'
end

function extend_shape(X::Array{Float64, 2}, batch_size::Int64, in_channels::Int64, height::Int64, width::Int64)
	Y = reshape(X, height, width, in_channels, batch_size)
	Y = permutedims(Y, [4, 3, 1, 2])
	return Y
end

function conv_layer(x, w, b, stride = (1, 1), padding = (0, 0))
	# x: (batch_size, in_channels, height, width)
	# w: (out_channels, in_channels, kernel_height, kernel_width)
	# b: (out_channels,)

	height, width = size(x)
	kernel_height, kernel_width = size(w)

	# Add padding to input data
	x = padarray(x, padding)

	# Reshape input data
	x = im2col(x, kernel_height, kernel_width)

	# Reshape weights
	w = reshape(w, (out_channels, in_channels * kernel_height * kernel_width))

	# Perform matrix multiplication
	y = w * x'

	# Add bias
	y .+= b'

	# Reshape output
	output_height = div(height + 2 * padding[1] - kernel_height, stride[1]) + 1
	output_width = div(width + 2 * padding[2] - kernel_width, stride[2]) + 1
	y = reshape(y, (out_channels, output_height, output_width, batch_size))
	y = permutedims(y, (4, 1, 2, 3))

	return y
end

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

function net(x, wh, b, wo, y)
	x = extend_shape(x, 1, 1, 10, 10)
	x̂ = conv_layer(x, wh, b)
	x̂.name = "x̂"
	ŷ = dense(wo, x̂, relu)
	ŷ.name = "ŷ"
	E = mean_squared_loss(y, ŷ)
	E.name = "loss"
	return topological_sort(E)
end

x = Variable(randn(10, 10), name = "x")
Wh = Variable(randn(10, 2), name = "wh")
Wo = Variable(randn(1, 10), name = "wo")
y = Variable([0.064], name = "y")
losses = Float64[]

graph = net(x, Wh, Wo, y)
forward!(graph)
backward!(graph)

for (i, n) in enumerate(graph)
	print(i, ". ")
	println(n)
end

