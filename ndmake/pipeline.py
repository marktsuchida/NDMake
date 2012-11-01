import collections
import functools
import re
import sys

from ndmake import depgraph
from ndmake import template
from ndmake import debug

dprint = debug.dprint_factory(__name__)

# A static representation of a computational pipeline.

# A depgraph.Graph is generated from a Pipeline.


Entity = collections.namedtuple("Entity", ("kind", "name", "entries"))
entity_kinds = {"global", "dimension", "subdomain", "dataset", "compute"}


class Pipeline:
    def __init__(self):
        self.entities = {} # (kind, name) -> Entity
        self.edges = set() # ((kind, name), (kind, name))
        self.last_global_entity_name = None

        self._sorted_keys = None

    def add_entity(self, entity):
        # Invalidate
        self._sorted_keys = None

        if (entity.kind, entity.name) in self.entities:
            raise KeyError("duplicate definition: {} {}".
                           format(entity.kind, entity.name))

        if entity.kind not in entity_kinds:
            raise KeyError("unknown entity kind: {}".format(entity.kind))

        preparer = "_prepare_{}_entity".format(entity.kind)
        entity = getattr(self, preparer)(entity)

        self.entities[(entity.kind, entity.name)] = entity

        processor = "_process_{}_entity".format(entity.kind)
        getattr(self, processor)(entity)

    def _add_edge(self, parent, child):
        self.edges.add((parent, child))

    def _check_identifier(self, name):
        return re.match(r"[A-Za-z_][0-9A-Za-z_]*", name) is not None

    def _dim_or_dom_key(self, name):
        items = name.split(".")
        if len(items) == 1:
            if self._check_identifier(name):
                return ("dimension", name)
        elif len(items) > 1:
            if all(map(self._check_identifier, items)):
                return ("subdomain", name)
        raise ValueError("invalid dimension or subdomain name: {}".
                         format(name))

    def _prepare_global_entity(self, entity):
        if entity.name is None:
            entity = Entity(entity.kind, "<default>", entity.entries)
        return entity

    def _process_global_entity(self, entity):
        # Make dependent on previous `global' entity so that order is
        # preserved in the later topological sort.
        if self.last_global_entity_name is not None:
            self._add_edge(("global", self.last_global_entity_name),
                           ("global", entity.name))
        self.last_global_entity_name = entity.name

    def _prepare_dimension_entity(self, entity):
        if not self._check_identifier(entity.name):
            raise ValueError("invalid dimension name: {}".format(entity.name))
        return entity

    def _process_dimension_entity(self, entity):
        _, name, entries = entity

        # Dimensions depend on the dimensions or domains of the parent scope.
        for dim_or_dom in entries.get("given", ()):
            self._add_edge(self._dim_or_dom_key(dim_or_dom),
                           ("dimension", name))

        mode = None
        mode_keys = ("values", "range", "command", "range_command", "match")
        for key in mode_keys:
            if key in entries:
                if mode is not None:
                    mode_keys_string = (", ".join(mode_keys[:-1]) + ", or " +
                                        mode_keys[-1])
                    raise KeyError("exactly one of {modes} must be given".
                                   format(modes=mode_keys_string))
                mode = key

        # Command-based surveyed dimensions optionally depend on inputs.
        if mode in ("command", "range_command"):
            for dataset in entries.get("inputs", ()):
                self._add_edge(("dataset", dataset), ("dimension", name))

        # Pattern-matching surveyed dimensions optionally depend on a compute.
        if mode == "match":
            if "producer" in entries:
                self._add_edge(("compute", entries["producer"]),
                               ("dimension", name))

    def _prepare_subdomain_entity(self, entity):
        # Check syntax.
        if self._dim_or_dom_key(entity.name)[0] != "subdomain":
            raise ValueError("invalid subdomain name: {}".format(entity.name))
        return entity

    def _process_subdomain_entity(self, entity):
        _, name, entries = entity

        # Subdomains depend on their superdomains.
        super_entity = self._dim_or_dom_key(".".join(name.split(".")[:-1]))
        self._add_edge(super_entity, ("subdomain", name))

        mode = None
        mode_keys = ("values", "range", "slice", "command",
                     "range_command", "slice_command", "match")
        mode_key_count = 0
        for key in mode_keys:
            if key in entries:
                if mode is not None:
                    mode_keys_string = (", ".join(mode_keys[:-1]) + ", or " +
                                        mode_keys[-1])
                    raise KeyError("exactly one of {modes} must be given".
                                   format(modes=mode_keys_string))
                mode = key

        # Command-based surveyed subdomains optionally depend on inputs.
        if mode in ("command", "range_command", "slice_command"):
            for dataset in entries.get("inputs", ()):
                self._add_edge(("dataset", dataset), ("dimension", name))

        # Pattern-matching surveyed subsomains optionally depend on a compute.
        if mode == "match":
            if "producer" in entries:
                self._add_edge(("compute", entries["producer"]),
                               ("dimension", name))

    def _prepare_dataset_entity(self, entity):
        if not self._check_identifier(entity.name):
            raise ValueError("invalid dataset name: {}".format(entity.name))
        return entity

    def _process_dataset_entity(self, entity):
        _, name, entries = entity

        if "filename" not in entries:
            raise KeyError("filename missing from dataset {}".format(name))

        # Datasets depend on the dimensions or domains of their scope.
        for dim_or_dom in entries.get("scope", ()):
            self._add_edge(self._dim_or_dom_key(dim_or_dom),
                           ("dataset", name))

        # Datasets optionally depend on a producer compute.
        if "producer" in entries:
            self._add_edge(("compute", entries["producer"]), ("dataset", name))

    def _prepare_compute_entity(self, entity):
        if not self._check_identifier(entity.name):
            raise ValueError("invalid compute name: {}".format(entity.name))
        return entity

    def _process_compute_entity(self, entity):
        _, name, entries = entity

        if "command" not in entries:
            raise KeyError("command missing from compute {}".format(name))

        # Computes depend on the dimensions or domains of their scope.
        for dim_or_dom in entries.get("scope", ()):
                self._add_edge(self._dim_or_dom_key(dim_or_dom),
                               ("compute", name))

        # Computes optionally depend on input datasets.
        for dataset in entries.get("inputs", ()):
            self._add_edge(("dataset", dataset), ("compute", name))

    @property
    def sorted_entity_keys(self):
        if self._sorted_keys is None:
            self._sorted_keys = self._topologically_sorted_entities_keys()
        return self._sorted_keys

    def _topologically_sorted_entities_keys(self):
        self._check_for_missing_entities()

        sorted_keys = []

        sources = set(self.entities)
        for parent, child in self.edges:
            sources.discard(child)

        visited = set()

        def visit(key, current_chain=()):
            if key in current_chain:
                cycle = (current_chain[current_chain.index(key):] +
                         (key,))
                raise ValueError("circular dependencies: " +
                                 " <- ".join("[{} {}]".format(*key)
                                             for key in cycle))
            if key not in visited:
                visited.add(key)
                for parent, child in self.edges:
                    if parent == key:
                        visit(child, current_chain + (key,))
                sorted_keys.append(key)

        for source in sources:
            visit(source)

        return reversed(sorted_keys)

    def _check_for_missing_entities(self):
        for parent, child in self.edges:
            for key in (parent, child):
                if key not in self.entities:
                    raise KeyError("missing {}: {}".format(*key))

    def write_graphviz(self, filename):
        self._check_for_missing_entities()

        def vertex_id(entity):
            return ".".join(entity).replace(".", "__")
        with open(filename, "w") as file:
            fprint = functools.partial(print, file=file)
            fprint("digraph entities {")
            for entity in self.entities:
                kind = self.entities[entity].kind
                label = " ".join(entity)
                shape = "box"
                color = "black"
                if kind == "compute":
                    shape = "box"
                    color = "black"
                elif kind == "dataset":
                    shape = "folder"
                    color = "navy"
                elif kind in ("dimension", "subdomain"):
                    shape = "box"
                    color = "red"
                elif kind == "global":
                    color = "darkgreen"
                fprint("{} [label=\"{}\" shape=\"{}\" color=\"{}\"];".
                       format(vertex_id(entity), label, shape, color))
            for parent, child in self.edges:
                parent_kind = self.entities[parent].kind
                child_kind = self.entities[child].kind
                dim_or_dom = ("dimension", "subdomain")
                color = "black"
                if parent_kind in dim_or_dom and child_kind not in dim_or_dom:
                    color = "gray"
                elif parent_kind in dim_or_dom and child_kind in dim_or_dom:
                    color = "red"
                elif parent_kind == "global" or child_kind == "global":
                    color = "darkgreen"
                fprint("{} -> {} [color=\"{}\"];".
                       format(vertex_id(parent), vertex_id(child), color))
            fprint("}")

    def depgraph(self):
        graph = depgraph.Graph()
        for kind, name in self.sorted_entity_keys:
            dprint("instantiating", "[{} {}]".format(kind, name))
            entity = self.entities[(kind, name)]
            getattr(self, "_instantiate_{}_entity".format(kind))(graph, entity)
        return graph

    def _get_extent_from_graph(self, graph, dim_or_dom_name):
        dim_name = dim_or_dom_name.split(".", 1)[0]
        dimension = graph.dimensions[dim_name]
        extent = dimension.extent_by_name(dim_or_dom_name)
        return extent

    def _add_vertex_to_graph(self, graph, vertex):
        dprint("adding vertex", vertex)
        graph.add_vertex(vertex)
        ancestoral_surveys = (extent.source
                              for extent in vertex.scope.extents
                              if extent.is_surveyed)
        for survey in ancestoral_surveys:
            self._add_edge_to_graph(graph, survey, vertex)

    def _add_edge_to_graph(self, graph, vertex1, vertex2):
        dprint("adding edge", vertex1, "to", vertex2)
        graph.add_edge(vertex1, vertex2)

    def _instantiate_global_entity(self, graph, entity):
        if "defs" in entity.entries:
            graph.templateset.append_global_defs(entity.entries["defs"])

    def _instantiate_dimension_or_domain_entity(self, graph, entity):
        kind, name, entries = entity
        new_template = graph.templateset.new_template

        if kind == "dimension":
            extents = list(self._get_extent_from_graph(graph, dim_or_dom)
                           for dim_or_dom in entries.get("given", ()))
            scope = depgraph.Space(extents)
        else:
            assert kind == "subdomain"
            dimension = graph.dimensions[name.split(".", 1)[0]]
            superextent_name, extent_name = name.rsplit(".", 1)
            superextent = dimension.extent_by_name(superextent_name)
            scope = superextent.scope

        survey = None

        if "values" in entries:
            classes = (depgraph.EnumeratedFullExtent,
                       depgraph.EnumeratedSubextent)
            tmpl = new_template("__values_{}".format(name), entries["values"])
            source = tmpl

        elif "range" in entries:
            classes = (depgraph.ArithmeticFullExtent,
                       depgraph.ArithmeticSubextent)
            tmpl = new_template("__range_{}".format(name), entries["range"])
            source = tmpl

        elif "slice" in entries:
            assert kind == "subdomain"
            classes = (None, depgraph.IndexedSubextent)
            tmpl = new_template("__slice_{}".format(name), entries["slice"])
            source = tmpl

        elif "command" in entries:
            classes = (depgraph.EnumeratedFullExtent,
                       depgraph.EnumeratedSubextent)
            tmpl = new_template("__vcmd_{}".format(name), entries["command"])
            if "transform" in entries:
                tfm_tmpl = new_template("__tform_{}".format(name),
                                        entries["transform"])
            else:
                tfm_tmpl = None
            surveyer = depgraph.ValuesCommandSurveyer(name, scope,
                                                      tmpl, tfm_tmpl)
            survey = depgraph.Survey(graph, name, scope, surveyer)
            self._add_vertex_to_graph(graph, survey)
            source = survey

        elif "range_command" in entries:
            classes = (depgraph.ArithmeticFullExtent,
                       depgraph.ArithmeticSubextent)
            tmpl = new_template("__rcmd_{}".format(name),
                                entries["range_command"])
            surveyer = depgraph.IntegerTripletCommandSurveyer(name, scope,
                                                              tmpl)
            survey = depgraph.Survey(graph, name, scope, surveyer)
            self._add_vertex_to_graph(graph, survey)
            source = survey

        elif "slice_command" in entries:
            assert kind == "subdomain"
            classes = (None, depgraph.IndexedSubextent)
            tmpl = new_template("__scmd_{}".format(name),
                                entries["slice_command"])
            surveyer = depgraph.IntegerTripletCommandSurveyer(name, scope,
                                                              tmpl)
            survey = depgraph.Survey(graph, name, scope, surveyer)
            self._add_vertex_to_graph(graph, survey)
            source = survey

        else:
            assert "match" in entries
            classes = (depgraph.EnumeratedFullExtent,
                       depgraph.EnumeratedSubextent)
            match_tmpl = new_template("__match_{}".format(name),
                                      entries["match"])
            if "transform" in entries:
                tfm_tmpl = new_template("__tform_{}".format(name),
                                        entries["transform"])
            else:
                tfm_tmpl = None
            surveyer = depgraph.FilenameSurveyer(name, scope,
                                                 match_tmpl, tfm_tmpl)
            survey = depgraph.Survey(graph, name, scope, surveyer)
            self._add_vertex_to_graph(graph, survey)
            source = survey
            
        if isinstance(source, depgraph.Survey):
            if isinstance(surveyer, depgraph.CommandSurveyer):
                for input in entries.get("inputs", ()):
                    dataset = graph.vertex_by_name(input, depgraph.Dataset)
                    self._add_edge_to_graph(graph, dataset, survey)

            if isinstance(surveyer, depgraph.FilenameSurveyer):
                if "producer" in entries:
                    compute = graph.vertex_by_name(entries["producer"],
                                                   depgraph.Computation)
                    self._add_edge_to_graph(graph, compute, survey)

        if kind == "dimension":
            dimension = depgraph.Dimension(name)
            extent = classes[0](dimension, scope, source)
            dimension.full_extent = extent
            graph.dimensions[name] = dimension # TODO Use a Graph method.
        else:
            extent = classes[1](superextent, extent_name, source)
            # TODO Use an Extent method.
            superextent.subextents[extent_name] = extent

    def _instantiate_dimension_entity(self, graph, entity):
        self._instantiate_dimension_or_domain_entity(graph, entity)

    def _instantiate_subdomain_entity(self, graph, entity):
        self._instantiate_dimension_or_domain_entity(graph, entity)

    def _instantiate_dataset_entity(self, graph, entity):
        _, name, entries = entity

        extents = list(self._get_extent_from_graph(graph, dim_or_dom)
                       for dim_or_dom in entries.get("scope", ()))
        scope = depgraph.Space(extents)

        tmpl = graph.templateset.new_template("__dataset_{}".format(name),
                                              entries["filename"])

        dataset = depgraph.Dataset(graph, name, scope, tmpl)
        self._add_vertex_to_graph(graph, dataset)

        if "producer" in entries:
            compute = graph.vertex_by_name(entries["producer"],
                                           depgraph.Computation)
            self._add_edge_to_graph(graph, compute, dataset)

    def _instantiate_compute_entity(self, graph, entity):
        _, name, entries = entity

        extents = list(self._get_extent_from_graph(graph, dim_or_dom)
                       for dim_or_dom in entries.get("scope", ()))
        scope = depgraph.Space(extents)

        tmpl = graph.templateset.new_template("__compute_{}".format(name),
                                              entries["command"])

        parallel = entries.get("parallel", "yes")
        try:
            if parallel == "yes":
                occupancy = 1
            elif parallel == "no":
                occupancy = sys.maxsize
            elif parallel.startswith("1/"):
                occupancy = int(parallel[2:])
                assert occupancy > 0
            else:
                assert false
        except:
            raise ValueError("invalid parallel specifier: {}")

        compute = depgraph.Computation(graph, name, scope, tmpl, occupancy)
        self._add_vertex_to_graph(graph, compute)

        for input in entity.entries.get("inputs", ()):
            dataset = graph.vertex_by_name(input, depgraph.Dataset)
            self._add_edge_to_graph(graph, dataset, compute)

