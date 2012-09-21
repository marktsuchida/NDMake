import configparser
from collections import OrderedDict

import depgraph
import template

def read_depgraph(graph, filename):
    parser = configparser.ConfigParser(allow_no_value=True,
                                       delimiters=(":",),
                                       comment_prefixes=("#",),
                                       strict=True, # Disallow duplicates.
                                       empty_lines_in_values=True,
                                       default_section="global",
                                       interpolation=None)
    parser.read(filename)

    for section in parser.sections():
        words = section.split()
        if not len(words):
            raise SyntaxError("empty section name")
        name = words[1] if len(words) == 2 else None
        if len(words) > 2:
            raise SyntaxError("too many words in section name")
        {"global": read_global_section,
         "dimension": read_dimension_section,
         "subdomain": read_subdomain_section,
         "dataset": read_dataset_section,
         "compute": read_compute_section}[words[0]](graph, name,
                                                    parser.items(section))

def read_global_section(graph, name, items):
    for key, value in items:
        if key == "defs":
            graph.templateset.append_global_defs(value)
        else:
            raise SyntaxError("unknown item `{}' in section [global {}]".
                              format(key, name))

def read_dimension_section(graph, name, items):
    items = OrderedDict(items)
    parent_domains = OrderedDict()
    parent_domain_names = items.pop("parents", "").split()
    for domain_name in parent_domain_names:
        dim_dom = domain_name.split(".")
        if len(dim_dom) == 1:
            dim = graph.dimensions[dim_dom[0]]
            dom = dim.full_domain
        elif len(dim_dom) == 2:
            dim = graph.dimensions[dim_dom[0]]
            dom = dim.domains[dim_dom[1]]
        parent_domains[dim] = dom
    parent_space = depgraph.Space(parent_domains)

    if "values" in items:
        values_tmpl = graph.templateset.new_template("__domain_values_" + name,
                                                     items.pop("values"))
        seq_source = depgraph.EnumeratedDomainSequenceSource(parent_space,
                                                             values_tmpl)
        if len(items):
            raise ValueError("unrecognized or mutually exclusive items in "
                             "section [dimension {}]".format(name))
    elif "range" in items:
        range_tmpl = graph.templateset.new_template("__domain_range_" + name,
                                                    items.pop("range"))
        seq_source = depgraph.RangeDomainSequenceSource(parent_space,
                                                        range_tmpl)
        if len(items):
            raise ValueError("unrecognized or mutually exclusive items in "
                             "section [dimension {}]".format(name))
    else:
        if "dataset" not in items:
            raise KeyError("one of `values', `range', or `dataset' required "
                           "in section [dimension {}]".format(name))
        dataset_name = items.pop("dataset")
        dataset = graph.vertex_by_name(dataset_name, depgraph.Dataset,
                                       allow_placeholder=True)

        if "range_command" in items:
            tmpl = graph.templateset.new_template("__domain_cmd" + name,
                                                  items.pop("range_command"))
            surveyer = depgraph.RangeCommandDomainSurveyer(name, parent_space,
                                                           tmpl)
            seq_source = depgraph.RangeDomainSequenceSource(parent_space,
                                                            surveyer)
            if len(items):
                raise ValueError("unrecognized or mutually exclusive items in "
                                 "section [dimension {}]".format(name))
        elif "values_command" in items:
            raise UnimplementedError()
        elif "extract" in items: # extract/transform (+ optional prefilter)
            raise UnimplementedError()

    dimension = depgraph.Dimension(name)
    domain = depgraph.FullDomain(seq_source, dimension)
    graph.dimensions[name] = dimension

def read_subdomain_section(graph, name, items):
    dimension, name = name.split(".")
    dimension = graph.dimensions[dimension]

    items = OrderedDict(items)
    parent_domains = OrderedDict()
    parent_domain_names = items.pop("parents", "").split()
    if parent_domain_names is None:
        parent_space = dimension.full_domain.space
    else:
        for domain_name in parent_domain_names:
            dim_dom = domain_name.split(".")
            if len(dim_dom) == 1:
                dim = graph.dimensions[dim_dom[0]]
                dom = dim.full_domain
            elif len(dim_dom) == 2:
                dim = graph.dimensions[dim_dom[0]]
                dom = dim.domains[dim_dom[1]]
            parent_domains[dim] = dom
        parent_space = depgraph.Space(parent_domains)
        if not parent_space.is_subspace(dimension.full_domain.space):
            raise ValueError("subdomain parent space must be subspace of full "
                             "domain parent space")

    subdomain_class = depgraph.SubsetDomain

    if "values" in items:
        values_tmpl = graph.templateset.new_template("__domain_values_" +
                                                     dimension.name + "." +
                                                     name,
                                                     items.pop("values"))
        seq_source = depgraph.EnumeratedDomainSequenceSource(parent_space,
                                                             values_tmpl)
        if len(items):
            raise ValueError("unrecognized or mutually exclusive items in "
                             "section [dimension {}]".format(name))
    elif "range" in items:
        range_tmpl = graph.templateset.new_template("__domain_range_" +
                                                    dimension.name + "." +
                                                    name,
                                                    items.pop("range"))
        seq_source = depgraph.RangeDomainSequenceSource(parent_space,
                                                        range_tmpl)
        if len(items):
            raise ValueError("unrecognized or mutually exclusive items in "
                             "section [dimension {}]".format(name))
    elif "slice" in items:
        slice_tmpl = graph.templateset.new_template("__domain_slice_" +
                                                    dimension.name + "." +
                                                    name,
                                                    items.pop("slice"))
        seq_source = depgraph.RangeDomainSequenceSource(parent_space,
                                                        slice_tmpl)
        subdomain_class = depgraph.SliceDomain
    else:
        if "dataset" not in items:
            raise KeyError("one of `values', `range', or `dataset' required "
                           "in section [dimension {}]".format(name))
        dataset_name = items.pop("dataset")
        dataset = graph.vertex_by_name(dataset_name, depgraph.Dataset,
                                       allow_placeholder=True)

        if "range_command" in items:
            tmpl = graph.templateset.new_template("__domain_cmd" + name,
                                                  items.pop("range_command"))
            surveyer = depgraph.RangeCommandDomainSurveyer(name, parent_space,
                                                           tmpl)
            seq_source = depgraph.RangeDomainSequenceSource(parent_space,
                                                            surveyer)
            if len(items):
                raise ValueError("unrecognized or mutually exclusive items in "
                                 "section [dimension {}]".format(name))
        elif "slice_command" in items:
            raise UnimplementedError()
        elif "values_command" in items:
            raise UnimplementedError()
        elif "extract" in items: # extract/transform (+ optional prefilter)
            raise UnimplementedError()

    domain = subdomain_class(seq_source, dimension, name)

def read_dataset_section(graph, name, items):
    items = OrderedDict(items)
    domains = OrderedDict()
    domain_names = items.pop("domains", "").split()
    for domain_name in domain_names:
        dim_dom = domain_name.split(".")
        if len(dim_dom) == 1:
            dim = graph.dimensions[dim_dom[0]]
            dom = dim.full_domain
        elif len(dim_dom) == 2:
            dim = graph.dimensions[dim_dom[0]]
            dom = tuple(dim.domains[d] for d in dim_dom[1].split("|"))
        domains[dim] = dom
    space = depgraph.Space(domains)
    
    tmpl = graph.templateset.new_template(name, items.pop("filename"))


    dataset = depgraph.Dataset(name, space, tmpl)
    graph.add_vertex(dataset)

    producer = items.pop("producer", None)
    if producer is not None:
        producer = graph.vertex_by_name(producer, depgraph.Computation,
                                        allow_placeholder=True)
        graph.add_edge(producer, dataset)

def read_compute_section(graph, name, items):
    items = OrderedDict(items)
    domains = OrderedDict()
    domain_names = items.pop("domains", "").split()
    for domain_name in domain_names:
        dim_dom = domain_name.split(".")
        if len(dim_dom) == 1:
            dim = graph.dimensions[dim_dom[0]]
            dom = dim.full_domain
        elif len(dim_dom) == 2:
            dim = graph.dimensions[dim_dom[0]]
            dom = tuple(dim.domains[d] for d in dim_dom[1].split("|"))
        domains[dim] = dom
    space = depgraph.Space(domains)

    tmpl = graph.templateset.new_template(name, items.pop("command"))
    
    parallel = items.pop("parallel", "no")
    # Note: we intensionally disallow YES, Yes, True, true, y, Y, t, T, etc.
    if parallel == "yes":
        parallel = True # Same as 1.0.
    elif parallel == "no":
        parallel = False # Serial execution.
    elif parallel[-1] == "%":
        parallel = float(parallel[:-1]) / 100.0 # Fraction of core count.
    else:
        parallel = int(parallel) # Number of processes.

    # TODO Resource groups to limit parallel execution

    compute = depgraph.Computation(name, space, tmpl, parallel)
    graph.add_vertex(compute)

    inputs = items.pop("input", "").split()
    inputs = list(graph.vertex_by_name(input, depgraph.Dataset,
                                       allow_placeholder=True)
                  for input in inputs)
    for input in inputs:
        graph.add_edge(input, compute)

