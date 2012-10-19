import argparse
import multiprocessing
import sys

import depgraph
import dispatch
import ndmakefile
import update


__doc__ = """Flexible automation for iterative computations."""


def run(argv):
    parser = argparse.ArgumentParser(prog=argv[0],
                                     description=__doc__)
    parser.add_argument("dataset_or_computation",
                        metavar="NAME", nargs="*",
                        help="datasets or computations to bring up to date "
                        "(default: all datasets and computations)")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-f", "--file", metavar="FILE",
                        default="NDMakefile")
    parser.add_argument("-j", "--jobs", metavar="N", type=int)
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("-L", "--check-symlink-times", action="store_true")
    parser.add_argument("-i", "--ignore-errors", action="store_true")
    parser.add_argument("-k", "--keep-going", action="store_true")
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument("-q", "--question", "--query", action="store_true")
    parser.add_argument("-s", "--silent", "--quiet", action="store_true")
    parser.add_argument("-t", "--touch", action="store_true")
    parser.add_argument("-o", "--assume-old",
                        metavar="DATASET_OR_COMPUTE")
    parser.add_argument("-W", "--assume-new",
                        metavar="DATASET_OR_COMPUTE")
    parser.add_argument("-g", "--write-graph", metavar="FILE")
    parser.add_argument("-G", "--write-section-graph", metavar="FILE")
    parser.add_argument("-d", "--debug", metavar="FLAGS")

    args = parser.parse_args()

    unimplemented_options = ["assume_new", "assume_old", "check_symlink_times",
                             "debug", "dry_run", "ignore_errors", "keep_going",
                             "question", "silent", "touch", "version"]
    for option in unimplemented_options:
        if vars(args)[option]:
            raise NotImplementedError("option not yet implemented: --{}".
                                      format(option.replace("_", "-")))

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

    if not args.jobs and args.parallel:
        args.jobs = multiprocessing.cpu_count()

    updater = update.Update(graph)
    dispatch.start_with_tasklet(updater.update_vertices(vertices_to_update,
                                                        jobs=args.jobs))


if __name__ == "__main__":
    run(sys.argv)

