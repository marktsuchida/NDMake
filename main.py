import depgraph
import ndmakefile
import update
import dispatch
import sys

# Just a test for now.

with open(sys.argv[1]) as file:
    graph = ndmakefile.NDMakefile(file).graph

updater = update.Update(graph)
try:
    dispatch.start_with_tasklet(updater.update_vertices(graph.sinks()))
except update.NotUpToDateException as e:
    vertex = e.args[0]
    if isinstance(vertex, depgraph.RuntimeDecorator):
        vertex = vertex.static_object
    print("not up to date:", vertex)

