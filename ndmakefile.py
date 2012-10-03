from collections import OrderedDict
import collections
import configparser

import depgraph
import template

# Parse an ndmake input file (an ndmakefile).
#
# The sections and entries of the configparser format (hybrid INI/RFC 822
# format) are first transformed into a graph of sections (Section objects) that
# describes the declarative dependencies between them (so that they can appear
# in arbitrary order in the input file). The section graph is topologically
# sorted before being transformed into a dependency graph (depgraph.StaticGraph
# and associated objects), so that we do not need to introduce placeholders in
# the depgraph.


DEBUG = True
if DEBUG:
    def dprint(*args):
        print("ndmakefile: {}:".format(args[0]), *args[1:])
else:
    def dprint(*args): pass


Section = collections.namedtuple("Section", ("kind", "name", "entries"))

class Error(Exception):
    def __init__(self, message, section_kind=None, section_name=None):
        super().__init__(message, section_kind, section_name)

    @property
    def message(self): return self.args[0]

    @property
    def kind(self): return self.args[1]

    @property
    def name(self): return self.args[2]

    def __str__(self):
        if self.kind is not None:
            if self.name is not None:
                return "{} {}: {}".format(self.kind, self.name, self.message)
            return "{}: {}".format(self.kind, self.message)
        return self.message

class ConsistencyError(Error): pass
class SyntaxError(Error): pass

class NDMakefile:
    section_kinds = {"global", "dimension", "subdomain", "dataset", "compute"}

    def __init__(self, file):
        parser = configparser.ConfigParser(allow_no_value=False,
                                           delimiters=(":",),
                                           comment_prefixes=("#",),
                                           strict=True, # Disallow duplicates.
                                           empty_lines_in_values=True,
                                           default_section="global",
                                           interpolation=None)
        parser.read_file(file)
        self.add_sections(parser)
        self.check_for_missing_sections()
        self.topologically_sort_sections()
        self.parse_sorted_sections()

    def add_sections(self, config_parser):
        # Keep a directed graph of sections.
        self.sections = {} # (kind, name) -> Section
        self.edges = [] # ((kind, parent_name), (kind, child_name))

        # Keep track of last-encountered `global' section, in order to preserve
        # their order of appearance.
        self.last_global_section = None

        for section in config_parser.sections():
            words = section.split()
            if not len(words):
                raise SyntaxError("missing section title")
            if len(words) > 2:
                raise SyntaxError("too many words in section title: `{}'".
                                  format(" ".join(words)))

            kind = words[0]
            if kind not in self.section_kinds:
                raise SyntaxError("unknown section type", kind)

            name = words[1] if len(words) == 2 else None
            entries = dict(config_parser.items(section))

            getattr(self, "add_{}_section".format(kind))(name, entries)

    def add_section(self, kind, name, entries):
        if not name:
            raise SyntaxError("missing name in section title", kind)
        if (kind, name) in self.sections:
            raise ConsistencyError("duplicate definition", kind, name)
        section = Section(kind, name, entries)
        self.sections[(kind, name)] = section
        return section

    def add_edge(self, parent, child):
        self.edges.append((parent, child))

    def check_scope(self, scope_specification, *, allow_multi_domain=False):
        # Yield (kind, name) where kind in ("dimension", "subdomain").
        # Full domains are yielded as "dimension"; subdomains as "subdomain".
        dims_and_doms = scope_specification.split()
        for dim_or_dom in dims_and_doms:
            dim_dom = dim_or_dom.split(".")
            if len(dim_dom) == 1:
                yield ("dimension", dim_dom[0])
            elif len(dim_dom) == 2:
                if not len(dim_dom[1]):
                    raise SyntaxError("invalid dimension specification")
                doms = dim_dom[1].split("|")
                if not allow_multi_domain and len(doms) != 1:
                    raise SyntaxError("invalid dimension specification")
                for dom in doms:
                    if dom == "":
                        yield ("dimension", dim_dom[0])
                    else:
                        yield ("subdomain", "{}.{}".format(dim_dom[0], dom))
            else:
                raise SyntaxError("invalid dimension specification")

    def add_global_section(self, name, entries):
        if name is None:
            name = "<default>"
        self.add_section("global", name, entries)

        remaining_keys = set(entries.keys())

        remaining_keys.discard("defs")
        if len(remaining_keys):
            raise SyntaxError("unknown {entry}: {}".
                              format(", ".join(remaining_keys),
                                     entry=("entry"
                                            if len(remaining_keys) == 1
                                            else "entries")),
                              "global", name)

        # Make dependent on previous `global' section so that order is
        # preserved in the later topological sort.
        if self.last_global_section is None:
            self.last_global_section = name
        else:
            self.add_edge(("global", self.last_global_section),
                          ("global", name))
            self.last_global_section = name

    def add_dimension_section(self, name, entries):
        self.add_section("dimension", name, entries)
        remaining_keys = set(entries.keys())

        # Dimensions depend on the dimensions or domains of the parent scope.
        remaining_keys.discard("parents")
        try:
            for dim_or_dom in self.check_scope(entries.get("parents", "")):
                self.add_edge(dim_or_dom, ("dimension", name))
        except SyntaxError as e:
            raise SyntaxError("parents: " + e.message, "dimension", name)
        
        mode_keys = ("values", "range", "values_command", "range_command",
                     "match")
        mode_key_count = 0
        for key in mode_keys:
            if key in remaining_keys:
                mode = key
                remaining_keys.remove(key)
                mode_key_count += 1
        if mode_key_count != 1:
            mode_keys_string = (", ".join(mode_keys[:-1]) + ", or " +
                                mode_keys[-1])
            raise SyntaxError("exactly one of {modes} must be given".
                              format(modes=mode_keys_string),
                              "dimension", name)

        # Command-based surveyed dimensions optionally depend on a dataset.
        if mode in ("values_command", "range_command"):
            if "input" in remaining_keys:
                remaining_keys.remove("input")
                self.add_edge(("dataset", entries["input"]),
                              ("dimension", name))

        # Pattern-matching surveyed dimensions optionally depend on a compute.
        if mode == "match":
            if "producer" in remaining_keys:
                remaining_keys.remove("producer")
                self.add_edge(("compute", entries["producer"]),
                              ("dimension", name))
            remaining_keys.discard("transform")

        if len(remaining_keys):
            raise SyntaxError("unknown or incompatible {entry}: {}".
                              format(", ".join(remaining_keys),
                                     entry=("entry"
                                            if len(remaining_keys) == 1
                                            else "entries")),
                              "dimension", name)

    def add_subdomain_section(self, name, entries):
        self.add_section("subdomain", name, entries)

        # Subdomains depend on their dimension.
        dim_dom = name.split(".")
        if len(dim_dom) != 2:
            raise SyntaxError("invalid name", "subdomain", name)
        dimension = dim_dom[0]
        self.add_edge(("dimension", dimension),
                      ("subdomain", name))

        remaining_keys = set(entries.keys())

        # Subdomains depend on the dimensions or domains of the parent scope.
        remaining_keys.discard("parents")
        try:
            for dim_or_dom in self.check_scope(entries.get("parents", "")):
                self.add_edge(dim_or_dom, ("subdomain", name))
        except SyntaxError as e:
            raise SyntaxError("parents: " + e.message, "subdomain", name)
        
        mode_keys = ("values", "range", "slice", "values_command",
                     "range_command", "slice_command", "match")
        mode_key_count = 0
        for key in mode_keys:
            if key in remaining_keys:
                mode = key
                remaining_keys.remove(key)
                mode_key_count += 1
        if mode_key_count != 1:
            mode_keys_string = (", ".join(mode_keys[:-1]) + ", or " +
                                mode_keys[-1])
            raise SyntaxError("exactly one of {modes} must be given".
                              format(modes=mode_keys_string),
                              "subdomain", name)

        # Command-based surveyed subdomains optionally depend on a dataset.
        if mode in ("values_command", "range_command", "slice_command"):
            if "input" in remaining_keys:
                remaining_keys.remove("input")
                self.add_edge(("dataset", entries["input"]),
                              ("dimension", name))

        # Pattern-matching surveyed subsomains optionally depend on a compute.
        if mode == "match":
            if "producer" in remaining_keys:
                remaining_keys.remove("producer")
                self.add_edge(("compute", entries["producer"]),
                              ("dimension", name))
            remaining_keys.discard("transform")

        if len(remaining_keys):
            raise SyntaxError("unknown or incompatible {entry}: {}".
                              format(", ".join(remaining_keys),
                                     entry=("entry"
                                            if len(remaining_keys) == 1
                                            else "entries")),
                              "subdomain", name)

    def add_dataset_section(self, name, entries):
        self.add_section("dataset", name, entries)

        remaining_keys = set(entries.keys())

        if "filename" not in remaining_keys:
            raise SyntaxError("`filename' missing", "dataset", name)
        remaining_keys.discard("filename")

        # Datasets depend on the dimensions or domains of their scope.
        remaining_keys.discard("dimensions")
        try:
            for dim_or_dom in self.check_scope(entries.get("dimensions", ""),
                                               allow_multi_domain=True):
                self.add_edge(dim_or_dom, ("dataset", name))
        except SyntaxError as e:
            raise SyntaxError("dimensions: " + e.message, "dataset", name)

        # Datasets optionally depend on a producer compute.
        if "producer" in remaining_keys:
            remaining_keys.remove("producer")
            self.add_edge(("compute", entries["producer"]),
                          ("dataset", name))

        if len(remaining_keys):
            raise SyntaxError("unknown {entry}: {}".
                              format(", ".join(remaining_keys),
                                     entry=("entry"
                                            if len(remaining_keys) == 1
                                            else "entries")),
                              "dataset", name)

    def add_compute_section(self, name, entries):
        self.add_section("compute", name, entries)

        remaining_keys = set(entries.keys())

        if "command" not in remaining_keys:
            raise SyntaxError("`command' missing", "compute", name)
        remaining_keys.discard("command")

        # Computes depend on the dimensions or domains of their scope.
        remaining_keys.discard("dimensions")
        try:
            for dim_or_dom in self.check_scope(entries.get("dimensions", ""),
                                               allow_multi_domain=True):
                self.add_edge(dim_or_dom, ("compute", name))
        except SyntaxError as e:
            raise SyntaxError("dimensions: " + e.message, "compute", name)

        # Computes optionally depend on input datasets.
        remaining_keys.discard("input")
        for input in entries.get("input", "").split():
            self.add_edge(("dataset", input),
                          ("compute", name))

        remaining_keys.discard("parallel")

        if len(remaining_keys):
            raise SyntaxError("unknown {entry}: {}".
                              format(", ".join(remaining_keys),
                                     entry=("entry"
                                            if len(remaining_keys) == 1
                                            else "entries")),
                              "compute", name)

    def check_for_missing_sections(self):
        for parent, child in self.edges:
            for section in (parent, child):
                if section not in self.sections:
                    raise ConsistencyError("missing", section[0], section[1])

    def topologically_sort_sections(self):
        sorted_sections = []

        sources = set(self.sections)
        for parent, child in self.edges:
            sources.discard(child)

        visited = set()

        def visit(section, current_chain=()):
            if section in current_chain:
                cycle = (current_chain[current_chain.index(section):] +
                         (section,))
                raise ConsistencyError("circular dependencies: " +
                                       " <- ".join("[{} {}]".format(*section)
                                                   for section in cycle))
            if section not in visited:
                visited.add(section)
                for parent, child in self.edges:
                    if parent == section:
                        visit(child, current_chain + (section,))
                sorted_sections.append(section)

        for source in sources:
            visit(source)

        self.sorted_sections = reversed(sorted_sections)

    def parse_sorted_sections(self):
        self.graph = depgraph.StaticGraph()
        for kind, name in self.sorted_sections:
            dprint("parsing", "[{} {}]".format(kind, name))
            section = self.sections[(kind, name)]
            getattr(self, "parse_{}_section".format(kind))(section)

    def parse_scope(self, scope_specification):
        # Return OrderedDict(Dimension -> tuple(Domain))
        dims_and_doms = scope_specification.split()
        scope = OrderedDict()
        for dim_or_dom in dims_and_doms:
            dim_dom = dim_or_dom.split(".")

            dim = self.graph.dimensions[dim_dom[0]]
            if dim in scope:
                raise ConsistencyError("duplicate dimension specification")

            if len(dim_dom) == 1:
                domains = (dim.full_domain,)

            elif len(dim_dom) == 2:
                doms = dim_dom[1].split("|")
                if len(doms) != len(set(doms)):
                    raise ConsistencyError("duplicate domain specification")

                domains = tuple((dim.domains[dom] if dom else dim.full_domain)
                                for dom in doms)

            scope[dim] = domains

        return scope

    def parse_global_section(self, section):
        if "defs" in section.entries:
            self.graph.templateset.append_global_defs(section.entries["defs"])

    def parse_dimension_or_domain_section(self, section):
        # Shorthand:
        new_template = self.graph.templateset.new_template

        name = section.name

        try:
            parent_scope = self.parse_scope(section.entries.get("parents", ""))
        except Error as e:
            raise e.__class__("parents: " + e.message, section.kind, name)
        parent_space = depgraph.Space(parent_scope)

        if "values" in section.entries:
            subdomain_class = depgraph.SubsetDomain
            tmpl = new_template("__values_{}".format(name),
                                section.entries["values"])
            seq_source = depgraph.EnumeratedSequenceSource(parent_space, tmpl)

        elif "range" in section.entries:
            subdomain_class = depgraph.SubsetDomain
            tmpl = new_template("__range_{}".format(name),
                                section.entries["range"])
            seq_source = depgraph.RangeSequenceSource(parent_space, tmpl)

        elif "slice" in section.entries:
            assert section.kind == "subdomain"
            subdomain_class = depgraph.SliceDomain
            tmpl = new_template("__slice_{}".format(name),
                                section.entries["slice"])
            seq_source = depgraph.RangeSequenceSource(parent_space, tmpl)

        elif "values_command" in section.entries:
            subdomain_class = depgraph.SubsetDomain
            tmpl = new_template("__vcmd_{}".format(name),
                                section.entries["values_command"])
            surveyer = depgraph.ValuesCommandDomainSurveyer(name, parent_space,
                                                            tmpl)
            self.graph.add_vertex(surveyer)
            seq_source = depgraph.EnumeratedSequenceSource(parent_space,
                                                           surveyer)

        elif "range_command" in section.entries:
            subdomain_class = depgraph.SubsetDomain
            tmpl = new_template("__rcmd_{}".format(name),
                                section.entries["range_command"])
            surveyer = depgraph.RangeCommandDomainSurveyer(name, parent_space,
                                                           tmpl)
            self.graph.add_vertex(surveyer)
            seq_source = depgraph.RangeSequenceSource(parent_space, surveyer)

        elif "slice_command" in section.entries:
            assert section.kind == "subdomain"
            subdomain_class = depgraph.SliceDomain
            tmpl = new_template("__scmd_{}".format(name),
                                section.entries["slice_command"])
            surveyer = depgraph.RangeCommandDomainSurveyer(name, parent_space,
                                                           tmpl)
            self.graph.add_vertex(surveyer)
            seq_source = depgraph.RangeSequenceSource(parent_space, surveyer)

        else:
            assert "match" in section.entries
            subdomain_class = depgraph.SubsetDomain
            match_tmpl = new_template("__match_{}".format(name),
                                      section.entries["match"])
            if "transform" in section.entries:
                tfm_tmpl = new_template("__tform_{}".format(name),
                                        section.entries["transform"])
            else:
                tfm_tmpl = None
            surveyer = depgraph.FilenamePatternDomainSurveyer(name,
                                                              parent_space,
                                                              match_tmpl,
                                                              tfm_tmpl)
            self.graph.add_vertex(surveyer)
            seq_source = depgraph.EnumeratedSequenceSource(parent_space,
                                                           surveyer)
            
        if "input" in section.entries:
            dataset = self.graph.vertex_by_name(section.entries["input"],
                                                depgraph.Dataset)
            self.graph.add_edge(dataset, surveyer)

        if "producer" in section.entries:
            compute = self.graph.vertex_by_name(section.entries["producer"],
                                                depgraph.Computation)
            self.graph.add_edge(compute, surveyer)

        if section.kind == "dimension":
            dimension = depgraph.Dimension(name)
            domain = depgraph.FullDomain(seq_source, dimension)
            self.graph.dimensions[name] = dimension
        else:
            dim_name, dom_name = name.split(".")
            dimension = self.graph.dimensions[dim_name]
            subdomain = subdomain_class(seq_source, dimension, dom_name)

    def parse_dimension_section(self, section):
        self.parse_dimension_or_domain_section(section)

    def parse_subdomain_section(self, section):
        self.parse_dimension_or_domain_section(section)

    def parse_dataset_section(self, section):
        try:
            scope = self.parse_scope(section.entries.get("dimensions", ""))
        except Error as e:
            raise e.__class__("dimensions: " + e.message,
                              "dataset", section.name)
        space = depgraph.Space(scope)

        tmpl = self.graph.templateset. \
                new_template("__dataset_{}".format(section.name), 
                             section.entries["filename"])

        dataset = depgraph.Dataset(section.name, space, tmpl)
        self.graph.add_vertex(dataset)

        if "producer" in section.entries:
            compute = self.graph.vertex_by_name(section.entries["producer"],
                                                depgraph.Computation)
            self.graph.add_edge(compute, dataset)

    def parse_compute_section(self, section):
        try:
            scope = self.parse_scope(section.entries.get("dimensions", ""))
        except Error as e:
            raise e.__class__("dimensions: " + e.message,
                              "computation", section.name)
        space = depgraph.Space(scope)

        tmpl = self.graph.templateset. \
                new_template("__compute_{}".format(section.name), 
                             section.entries["command"])

        parallel = section.entries.get("parallel", "no")
        try:
            if parallel == "yes":
                parallel = True # Same as 1.0.
            elif parallel == "no":
                parallel = False # Serial execution.
            elif parallel[-1] == "%":
                # Fraction of core count.
                parallel = float(parallel[:-1]) / 100.0
            else:
                # Number of processes.
                parallel = int(parallel)
        except:
            raise SyntaxError("invalid `parallel' specifier",
                              "compute", section.name)

        compute = depgraph.Computation(section.name, space, tmpl, parallel)
        self.graph.add_vertex(compute)

        for input in section.entries.get("input", "").split():
            dataset = self.graph.vertex_by_name(input, depgraph.Dataset)
            self.graph.add_edge(dataset, compute)

