import argparse
import sys

from ndmake import debug
from ndmake import depgraph
from ndmake import dispatch
from ndmake import parse
from ndmake import space


__doc__ = """Flexible automation for iterative computations."""


def run(argv=sys.argv):
    parser = argparse.ArgumentParser(prog=argv[0],
                                     description=__doc__)
    parser.add_argument("targets",
                        metavar="TARGET", nargs="*",
                        help="dataset or computation to bring up to date "
                        "(default: all datasets and computations)")

    parser.add_argument("-V", "--version", action="store_true",
                        help="display version number and quit")

    parser.add_argument("-f", "--file", metavar="FILE",
                        default="NDMakefile",
                        help="read the pipeline from FILE (default: "
                        "%(default)s)")

    parser.add_argument("-p", "--parallel", action="store_true",
                        help="run commands concurrently (implies --jobs=NCPU "
                        "if --jobs is not given, where NCPU is the number of "
                        "CPU cores)")
    parser.add_argument("-j", "--jobs", metavar="N", type=int,
                        help="maximum number of commands to run concurrently "
                        "(implies --parallel)")

    parser.add_argument("-L", "--check-symlink-times", action="store_true",
                        help="[NOT IMPLEMENTED] check the modification times "
                        "of symbolic links instead of their targets")
    parser.add_argument("-C", "--directory", metavar="DIR",
                        help="change the working directory to DIR before "
                        "running")
    # TODO Might want to add options to set environment variables for command
    # execution (useful e.g. for PYTHONPATH if using a different Python version
    # to run computations).

    parser.add_argument("-D", "--depth-first", action="store_true",
                        help="[NOT IMPLEMENTED] start descendant computations "
                        "as soon as possible, instead of waiting for parents "
                        "to complete")

    parser.add_argument("-i", "--ignore-errors", action="store_true",
                        help="[NOT IMPLEMENTED] ignore nonzero exit statuses "
                        "returned by commands (so long as the expected output "
                        "files are produced)")
    parser.add_argument("-k", "--keep-going", action="store_true",
                        help="[NOT IMPLEMENTED] don't stop at the first "
                        "error; continue running while there are computations "
                        "available for update")

    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="[NOT IMPLEMENTED] don't actually run any "
                        "computations, but print what would be run (commands "
                        "for computed dimensions and subdomains may be run)")
    parser.add_argument("-q", "--question", "--query", action="store_true",
                        help="[NOT IMPLEMENTED] determine whether the given "
                        "targets are up to date and return an exit status of "
                        "0 if they are; 1 otherwise")

    parser.add_argument("-v", "--verbose", "--print-commands",
                        action="store_true",
                        help="print each command before running")
    parser.add_argument("-s", "--silent", "--quiet", action="store_true",
                        help="don't print target names as the dependency "
                        "graph is traversed")

    parser.add_argument("-c", "--cache", "--save-cache", action="store_true",
                        help="cache file modification times and computation "
                        "up-to-date statuses (speeds up reruns when there are "
                        "a large number of files, but will not detect changes "
                        "unless the cache is explicitly cleared)")

    parser.add_argument("--clear-cache", action="store_true",
                        help="clear the modification time cache for the "
                        "given targets (and all of their descendants)")
    parser.add_argument("--clean", "--remove-files", action="store_true",
                        help="remove the files belonging to the given targets")

    # XXX --clean-outdated does not use the previously cached mtimes but does
    # delete them, regardless of whether or not --cache is given. (Implement by
    # first clearing the cache for each vertex.)
    parser.add_argument("--clean-outdated", action="store_true",
                        help="[NOT IMPLEMENTED] remove all outdated files")

    parser.add_argument("-t", "--touch", action="store_true",
                        help="[NOT IMPLEMENTED] without running commands, "
                        "update only the modification time for the files "
                        "belonging to the given targets (and all of their "
                        "ancestors), so that they will be considered up to "
                        "date the next time ndmake is invoked")
    parser.add_argument("--touch-cache", action="store_true",
                        help="[NOT IMPLEMENTED] without touching files, mark "
                        "the given targets (and all of their ancestors) "
                        "as being up to date in the modification time cache")

    # XXX --assume-old=dataset needs special check: the producing computation
    # must not be slated to be executed due to other outputs being outdated.
    # --assume-old=compute is equivalent to specifying --assume-old on each of
    # the computation's outputs.
    parser.add_argument("-o", "--assume-old", action="append",
                        metavar="TARGET",
                        help="[NOT IMPLEMENTED] don't recompute TARGET, and "
                        "don't recompute any of its descendants if the only "
                        "reason for doing so would be because they are older "
                        "than TARGET")
    # XXX --assume-new=compute is equivalent to specifying --assume-new on each
    # of the computation's outputs.
    parser.add_argument("-W", "--assume-new", action="append",
                        metavar="TARGET",
                        help="[NOT IMPLEMENTED] don't recompute TARGET, but "
                        "force all of its descendants to be recomputed")

    parser.add_argument("-g", "--write-graph", metavar="FILE",
                        help="write the dependency graph to FILE (in "
                        "GraphViz .dot format)")
    parser.add_argument("-G", "--write-pipeline", metavar="FILE",
                        help="write the static pipeline graph (a "
                        "representation of the ndmakefile) to FILE (in "
                        "GraphViz .dot format) (mostly for debugging)")
    parser.add_argument("-d", "--debug", metavar="CATEGORIES",
                        help="turn on debug output; currently allowed "
                        "CATEGORIES (comma-separated) are: {}".
                        format(", ".join(debug.categories())))

    args = parser.parse_args()

    if args.version:
        print("this is an early alpha version of ndmake")
        sys.exit(0)

    unimplemented_options = [
                             "check_symlink_times",
                             "directory",
                             "depth_first",
                             "ignore_errors",
                             "keep_going",
                             "dry_run",
                             "question",
                             "clean_outdated",
                             "touch",
                             "touch_cache",
                             "assume_old",
                             "assume_new",
                            ]
    for option in unimplemented_options:
        if vars(args)[option]:
            raise NotImplementedError("option not yet implemented: --{}".
                                      format(option.replace("_", "-")))

    if args.debug:
        for category in (c.lower().strip() for c in args.debug.split(",")):
            debug.enable_dprint(category)

    with open(args.file) as file:
        pipeline = parse.read_ndmakefile(file)

    if args.write_pipeline:
        pipeline.write_graphviz(args.write_pipeline)
        if not args.write_graph:
            sys.exit(0)

    graph = pipeline.depgraph()

    if args.write_graph:
        graph.write_graphviz(args.write_graph)
        sys.exit(0)

    target_vertices = []
    for vertex_name in args.targets:
        if ":" in vertex_name:
            type_, vertex_name = vertex_name.split(":", 1)
            type_ = depgraph.Computation if type_ == "c" else depgraph.Dataset
            vertex = graph.vertex_by_name(vertex_name, type_)
            target_vertices.append(vertex)
        else:
            try:
                vertex = graph.vertex_by_name(vertex_name, depgraph.Dataset)
            except KeyError:
                vertex = None
            if vertex:
                target_vertices.append(vertex)
                continue
            try:
                vertex = graph.vertex_by_name(vertex_name, depgraph.Computation)
            except KeyError:
                raise KeyError("no dataset or computation named {}".
                               format(vertex_name))
            else:
                target_vertices.append(vertex)

    options = {}

    if args.clean or args.clear_cache: # Clean mode.
        if not target_vertices:
            raise NotImplementedError("--clean without targets is not "
                                      "implemented yet (in the future, it "
                                      "will clean all targets)")

        # First, load surveyed extents by traversing the depgraph but skipping
        # datasets and computations.
        options["survey_only"] = True
        options["cache"] = False
        options["print_executed_commands"] = args.verbose
        options["print_traversed_vertices"] = False
        tasklet = graph.update_vertices(graph.sinks(), options)
        dispatch.start_with_tasklet(tasklet)

        # Now do the cleaning.
        for vertex in target_vertices:
            vertex.clean(graph, space.Element(), cache_only=not args.clean)

        return

    # Standard update mode.
    if not target_vertices:
        target_vertices = graph.sinks()

    options["cache"] = args.cache

    if args.jobs is not None:
        if args.jobs < 1:
            raise ValueError("--jobs argument must be at least 1 (got {:d})".
                             format(args.jobs))
        if args.jobs > 1:
            args.parallel = True

    if args.parallel:
        options["parallel"] = True
        if args.jobs is not None:
            options["jobs"] = args.jobs

    options["print_executed_commands"] = args.verbose
    options["print_traversed_vertices"] = not args.silent

    tasklet = graph.update_vertices_with_threadpool(target_vertices, options)
    dispatch.start_with_tasklet(tasklet)


if __name__ == "__main__":
    run(sys.argv)

