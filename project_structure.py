from graphviz import Digraph

dot = Digraph()
dot.attr(rankdir="LR")

dot.node("A", "Project Root")
dot.node("B", ".devcontainer/devcontainer.json")
dot.node("C", ".gitignore")
dot.node("D", ".npmignore")
dot.node("E", "readme.md")
dot.node("F", "src")
dot.node("G", "reports")
dot.node("H", "automatic_differentiation")
dot.node("I", "reference_solution")
dot.node("J", "test-project")
dot.node("K", "basic_structures.jl")
dot.node("L", "backward_pass.jl")
dot.node("M", "forward_pass.jl")
dot.node("N", "graph_building.jl")
dot.node("O", "scalar_operators.jl")
dot.node("P", "broadcasted_operators.jl")
dot.node("Q", "convolution.jl")
dot.node("R", "in_tensorflow.py")
dot.node("S", "in_pytorch.py")
dot.node("T", "config.py")
dot.node("U", "mnist_dataset.py")
dot.node("V", "notes.txt")
dot.node("W", "results_pytorch.txt")
dot.node("X", "results_tensorflow.txt")
dot.node("Y", "test-utils-no-lc.sh")
dot.node("Z", "test.sh")

edges = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("A", "E"),
    ("A", "F"),
    ("A", "G"),
    ("F", "H"),
    ("F", "I"),
    ("F", "J"),
    ("H", "K"),
    ("H", "L"),
    ("H", "M"),
    ("H", "N"),
    ("H", "O"),
    ("H", "P"),
    ("H", "Q"),
    ("I", "R"),
    ("I", "S"),
    ("I", "T"),
    ("I", "U"),
    ("I", "V"),
    ("I", "W"),
    ("I", "X"),
    ("J", "Y"),
    ("J", "Z"),
]

for edge in edges:
    dot.edge(*edge)

dot.render("project_structure.gv", view=True)
