import depgraph
import ndmakefile
import update
import dispatch
import sys

# Just a test for now:
graph = depgraph.StaticGraph()
ndmakefile.read_depgraph(graph, sys.argv[1])

update = update.Update(graph, update.isuptodate)
dispatch.start_with_tasklet(update.update_sinks())

