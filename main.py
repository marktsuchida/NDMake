import depgraph
import ndmakefile
import sys

# Just a test for now:
graph = depgraph.StaticGraph()
ndmakefile.read_depgraph(graph, sys.argv[1])

