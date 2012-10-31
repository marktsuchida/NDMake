import argparse
import multiprocessing
import sys

from ndmake import debug
from ndmake import depgraph
from ndmake import dispatch
from ndmake import ndmakefile


__doc__ = """Flexible automation for iterative computations."""


def run(argv=sys.argv):
    parser = argparse.ArgumentParser(prog=argv[0],
                                     description=__doc__)
    parser.add_argument("dataset_or_computation",
                        metavar="NAME", nargs="*",
                        help="datasets or computations to bring up to date "
                        "(default: all datasets and computations)")
    parser.add_argument("-V", "--version", action="store_true",
                        help="not implemented")
    parser.add_argument("-f", "--file", metavar="FILE",
                        default="NDMakefile",
                        help="dependency graph input file "
                        "(default: NDMakefile)")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="run commands concurrently (implies --jobs=NCPU "
                        "if --jobs is not given, where NCPU is the number of "
                        "CPU cores)")
    parser.add_argument("-j", "--jobs", metavar="N", type=int,
                        help="maximum number of commands to run concurrently "
                        "(implies --parallel)")
    parser.add_argument("-L", "--check-symlink-times", action="store_true",
                        help="not implemented")
    parser.add_argument("-i", "--ignore-errors", action="store_true",
                        help="not implemented")
    parser.add_argument("-k", "--keep-going", action="store_true",
                        help="not implemented")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="not implemented")
    parser.add_argument("-q", "--question", "--query", action="store_true",
                        help="not implemented")
    parser.add_argument("-v", "--verbose", "--print-commands",
                        action="store_true",
                        help="not implemented")
    parser.add_argument("-s", "--silent", "--quiet", action="store_true",
                        help="not implemented")
    parser.add_argument("-c", "--clean-outdated", action="store_true",
                        help="not implemented")
    parser.add_argument("--clean", "--clean-only", action="store_true",
                        help="not implemented")
    parser.add_argument("--clean-onward", action="store_true",
                        help="not implemented")
    parser.add_argument("-T", "--touch-outdated", action="store_true",
                        help="not implemented")
    parser.add_argument("-t", "--touch", "--touch-only", action="store_true",
                        help="not implemented")
    parser.add_argument("--touch-upto", action="store_true",
                        help="not implemented")
    parser.add_argument("-o", "--assume-old", action="append",
                        metavar="DATASET_OR_COMPUTE",
                        help="not implemented")
    parser.add_argument("-W", "--assume-new", action="append",
                        metavar="DATASET_OR_COMPUTE",
                        help="not implemented")
    parser.add_argument("-g", "--write-graph", metavar="FILE",
                        help="write dependency graph in GraphViz dot format "
                        "to FILE")
    parser.add_argument("-G", "--write-section-graph", metavar="FILE",
                        help="write input file section dependency graph in "
                        "GraphViz dot format to FILE (mostly for debugging)")
    parser.add_argument("-d", "--debug", metavar="CATEGORIES",
                        help="currently allowed CATEGORIES (comma-separated) "
                        "are: {}".format(", ".join(debug.categories())))

    args = parser.parse_args()

    unimplemented_options = ["assume_new", "assume_old", "check_symlink_times",
                             "dry_run", "ignore_errors", "keep_going",
                             "question", "silent", "touch", "version"]
    for option in unimplemented_options:
        if vars(args)[option]:
            raise NotImplementedError("option not yet implemented: --{}".
                                      format(option.replace("_", "-")))

    if args.debug:
        for category in (c.lower().strip() for c in args.debug.split(",")):
            debug.enable_dprint(category)

    with open(args.file) as file:
        input_file = ndmakefile.NDMakefile(file)

    if args.write_graph:
        input_file.graph.write_graphviz(args.write_graph)
    if args.write_section_graph:
        input_file.write_sections_graphviz(args.write_section_graph)
    if args.write_graph or args.write_section_graph:
        sys.exit(0)

    graph = input_file.graph

    vertices_to_update = []
    for vertex_name in args.dataset_or_computation:
        if ":" in vertex_name:
            type_, vertex_name = vertex_name.split(":", 1)
            type_ = depgraph.Computation if type_ == "c" else depgraph.Dataset
            vertex = graph.vertex_by_name(vertex_name, type_)
            vertices_to_update.append(vertex)
        else:
            try:
                vertex = graph.vertex_by_name(vertex_name, depgraph.Dataset)
            except KeyError:
                vertex = None
            if vertex:
                vertices_to_update.append(vertex)
                continue
            try:
                vertex = graph.vertex_by_name(vertex_name, depgraph.Computation)
            except KeyError:
                raise KeyError("no dataset or computation named {}".
                               format(vertex_name))
            else:
                vertices_to_update.append(vertex)

    if not vertices_to_update:
        vertices_to_update = graph.sinks()

    if args.parallel and not args.jobs:
        args.jobs = multiprocessing.cpu_count()
    if args.jobs is not None and args.jobs < 1:
        raise ValueError("--jobs argument must be at least 1 (got {:d})".
                         format(args.jobs))
    if args.jobs is not None and args.jobs > 1:
        args.parallel = True

    tasklet = graph.update_vertices_with_threadpool(vertices_to_update,
                                                    parallel=args.parallel,
                                                    jobs=args.jobs)
    dispatch.start_with_tasklet(tasklet)


if __name__ == "__main__":
    run(sys.argv)

