import itertools
import collections
from collections import OrderedDict
import shlex
import template

# Static data representation and dynamic execution context are partially
# separated: Instances of StaticGraph and individual model objects do not store
# any runtime state, but some methods of the model objects do act on runtime
# context. Such runtime data is stored in a DynamicGraph instance, which is
# passed around to all runtime methods. In other words, double dispatch is used
# with the model object as primary target and runtime context (dynamic_graph)
# as secondary target.

# A documentation-only decorator.
def runtime(method): return method


#
# Graph facade
#

class StaticGraph:
    # The StaticGraph object is the facade to the DAC of Vertex objects.
    #
    # Vertices and edges are managed by integer vertex ids. The Vertex objects
    # themselves do not hold pointers to each other. This arrangement, in
    # addition to permitting whole-graph analysis, makes it easier to manage
    # runtime information attached to each vertex.
    #
    # The acyclic nature of the directed graph is enforced during graph
    # construction. XXX Might be more efficient to check later.
    #
    # The graph is a pure representation of an ndmake configuration and does
    # not hold runtime information (such as up-to-date state).
    #
    # Although not an intended use, a single Vertex can safely belong to
    # multiple StaticGraphs, as Vertex objects are immutable and agnostic of
    # their connections.

    def __init__(self):
        self._vertex_id_generator = itertools.count(0)

        # Vertices.
        self._vertex_id_map = {} # Vertex -> id
        self._id_vertex_map = {} # id -> Vertex
        self._name_id_map = {} # (name, type_) -> id

        # Edges.
        self._parents = {} # id -> set(ids)
        self._children = {} # id -> set(ids)

        # Dimensions.
        self.dimensions = {} # name -> Dimension

        # Templates.
        self.templateset = template.TemplateSet()

    def check_consistency(self):
        for vertex in self._vertex_id_map.keys():
            if isinstance(vertex, VertexPlaceholder):
                raise TypeError("placeholder remains in graph")

    def vertex_by_name(self, name, type_, allow_placeholder=False):
        """Return a vertex with the given name and type.

        Optionally, add a vertex placeholder with the given name and type.
        The placeholder can later be replaced with the real vertex.
        """
        if allow_placeholder and (name, type_) not in self._name_id_map:
            placeholder = VertexPlaceholder(name, type_)
            self.add_vertex(placeholder)

        try:
            vertex_id = self._name_id_map[(name, type_)]
        except KeyError:
            raise KeyError("no {} vertex named {}".format(type_.__name__,
                                                          name))
        return self._id_vertex_map[vertex_id]

    def add_vertex(self, vertex):
        """Add an isolated vertex to the graph and return the vertex id."""
        name_key = (vertex.name, vertex.namespace_type)
        if name_key in self._name_id_map:
            existing_id = self._name_id_map[name_key]
            existing_vertex = self._id_vertex_map[existing_id]
            if isinstance(existing_vertex, VertexPlaceholder):
                self._id_vertex_map[existing_id] = vertex
                return existing_id

        vertex_id = next(self._vertex_id_generator)
        self._vertex_id_map[vertex] = vertex_id
        self._id_vertex_map[vertex_id] = vertex
        self._name_id_map[name_key] = vertex_id
        return vertex_id

    def _vertex_id(self, vertex, add_if_not_member=False):
        if add_if_not_member and vertex not in self._vertex_id_map:
            return self.add_vertex(vertex)
        return self._vertex_id_map[vertex]

    def add_edge(self, from_vertex, to_vertex):
        """Add an edge between two vertices.

        If either or both of the vertices do not belong to the graph, add them
        as well.
        """
        from_id = self._vertex_id(from_vertex, add_if_not_member=True)
        to_id = self._vertex_id(to_vertex, add_if_not_member=True)

        if from_id == to_id:
            raise ValueError("attempt to create self-dependent vertex")
        if self._is_ancestor(to_vertex, from_vertex):
            raise ValueError("attmpt to create cycle in graph")

        self._parents.setdefault(to_id, set()).add(from_id)
        self._children.setdefault(from_id, set()).add(to_id)

    def _is_ancestor(self, the_vertex, other_vertex):
        # Not the most efficient traversal, but good enough for now.
        for child in self.children_of(the_vertex):
            if child is other_vertex or self._is_ancestor(child, other_vertex):
                return True

    def parents_of(self, vertex):
        """Return the parent vertices of the given vertex."""
        return list(self._id_vertex_map[i] for i in
                    self._parents.get(self._vertex_id_map[vertex], []))

    def children_of(self, vertex):
        """Return the child vertices of the given vertex."""
        return list(self._id_vertex_map[i] for i in
                    self._children.get(self._vertex_id_map[vertex], []))

    def sources(self):
        """Return all vertices that do not have parents.

        Includes isolated vertices, if any.
        """
        return list(self._id_vertex_map[id]
                    for id in self._id_vertex_map.keys()
                    if not len(self._parents.setdefault(id, set())))

    def sinks(self):
        """Return all vertices that do not have children.

        Includes isolated vertices, if any.
        """
        return list(self._id_vertex_map[id]
                    for id in self._id_vertex_map.keys()
                    if not len(self._children.setdefault(id, set())))


class DynamicGraph:
    def __init__(self, static_graph, runtime_proxy_factory=lambda x, y: x):
        self.static_graph = static_graph

        if isinstance(runtime_proxy_factory, collections.Mapping):
            def factory(graph, object):
                for type_, proxy_type in runtime_proxy_factory.items():
                    if isinstance(object, type_):
                        return proxy_type(graph, object)
                if Ellipsis in runtime_proxy_factory:
                    return runtime_proxy_factory[Ellipsis](graph, object)
                return object
        else:
            factory = runtime_proxy_factory
        self.runtime_proxy_factory = factory

        self.runtime_proxies = {} # object -> proxy

    _forwarded_attrs = ("dimensions", "templateset",)
    def __getattr__(self, name):
        if name in _forwarded_attrs:
            return getattr(self.static_graph, name)

    def vertex_by_name(self, name, type_):
        vertex = self.static_graph.vertex_by_name(name, type_)
        return self.runtime(vertex)

    def parents_of(self, vertex):
        if isinstance(vertex, RuntimeProxy):
            vertex = vertex.static_object
        parents = self.static_graph.parents_of(vertex)
        return list(self.runtime(v) for v in parents)

    def children_of(self, vertex):
        if isinstance(vertex, RuntimeProxy):
            vertex = vertex.static_object
        children = self.static_graph.children_of(vertex)
        return list(self.runtime(v) for v in children)

    def sources(self):
        sources = self.static_graph.sources()
        return list(self.runtime(v) for v in sources)

    def sinks(self):
        sinks = self.static_graph.sinks()
        return list(self.runtime(v) for v in sinks)

    def runtime(self, object):
        # Return the runtime proxy object for the given object.

        if isinstance(object, RuntimeProxy):
            return object

        if object not in self.runtime_proxies:
            self.runtime_proxies[object] = self.runtime_proxy_factory(self,
                                                                      object)
        return self.runtime_proxies[object]


#
# Runtime proxy objects
#

class RuntimeProxy:
    # We keep this simple, rather than dealing with various edge cases upfront.

    def __init__(self, dynamic_graph, static_object):
        self.graph = dynamic_graph
        self.static_object = static_object

    def __str__(self):
        return "<{} {}>".format(self.__class__.__name__,
                                str(self.static_object))

    def __getattr__(self, name):
        return getattr(self.static_object, name)


class VertexStateRuntime(RuntimeProxy):
    def __init__(self, graph, vertex):
        super().__init__(graph, vertex)

        self.mtime = None
        self.is_up_to_date = None # None = unknown; False = known out of date.


#
# Spatial
#

class Spatial:
    # Anything that has an associated space.
    @property
    def space(self):
        assert False, "abstract method call"


#
# Vertex
#

class Vertex(Spatial):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "<{} \"{}\">".format(type(self).__name__, self.name)

    @property
    def namespace_type(self):
        if isinstance(self, Dataset):
            return Dataset
        if isinstance(self, Computation):
            return Computation
        if isinstance(self, DomainSurveyer):
            return DomainSurveyer

    def is_locally_up_to_date(self, dynamic_graph):
        assert False, "abstract method call"

    def compute(self, dynamic_graph):
        assert False, "abstract method call"


class VertexPlaceholder(Vertex):
    def __init__(self, name, type_):
        super().__init__(name)
        self.type_ = type_

    def __str__(self):
        return "<{} placeholder \"{}\">".format(self.type_.__name__, self.name)

    @property
    def namespace_type(self):
        return self.type_


#
# Dimension
#

class Dimension:
    def __init__(self, name):
        self.name = name
        self.domains = {} # name -> domain

    @property
    def full_domain(self):
        return self.domains[Ellipsis]


#
# Domain sequence sources
#

# Sequence sources supply the range or value list for a domain. Sequence
# sources can be configured with static ranges or value lists, or can use a
# domain surveyer at runtime to determine the range or value list.

class DomainSequenceSource(Spatial):
    def __init__(self, parent_space):
        self._space = parent_space

    @property
    def space(self):
        return self._space

    @runtime
    def sequence(self, dynamic_graph, parent_point=Ellipsis):
        assert False, "abstract method call"

class EnumeratedDomainSequenceSource(DomainSequenceSource):
    def __init__(self, parent_space, values):
        super().__init__(parent_space)
        self.values = values # Surveyer or template.

    @runtime
    def sequence(self, dynamic_graph, parent_point=Ellipsis):
        if isinstance(self, values, DomainSurveyer):
            # Dynamic domain.
            surveyer = self.values
            return surveyer.values(dynamic_graph, parent_point)

        # Static domain.
        template = self.values
        dict_ = self.space.point_to_dict(parent_point)
        filename_expander = self.space.filename_expander(dynamic_graph)
        rendered_values = template.render(dict_, filename_expander)
        return shlex.split(rendered_values)

class RangeDomainSequenceSource(DomainSequenceSource):
    def __init__(self, parent_space, rangeargs):
        super().__init__(parent_space)
        self.rangeargs = rangeargs # Surveyer or template.

    @runtime
    def sequence(self, dynamic_graph, parent_point=Ellipsis):
        if isinstance(self.rangeargs, DomainSurveyer):
            # Dynamic domain.
            surveyer = self.rangeargs
            values = surveyer.values(dynamic_graph, parent_point)
            if isinstance(values, range):
                return values
            values = sorted(int(v) for v in values)

            if len(values) == 0:
                return range(0)

            if len(values) == 1:
                return range(values[0], values[0] + 1)

            start = values[0]
            stop = values[-1] + 1
            step = values[1] - values[0]
            range_ = range(start, stop, step)
            for ref, obs in zip(range_, values):
                if obs != ref:
                    raise ValueError("surveyed domain values do not "
                                     "constitute a range")
            return range_

        # Static domain.
        template = self.values
        dict_ = self.space.point_to_dict(parent_point)
        filename_expander = self.space.filename_expander(dynamic_graph)
        rendered_values = template.render(dict_, filename_expander)
        rangeargs = tuple(int(a) for a in rendered_values.split())
        if len(rangeargs) not in range(1, 4):
            raise ValueError("range template must expand to 1-3 integers")
        return range(*rangeargs)


#
# Domain surveyer
#

# Domain surveyers dynamically determine a domain at runtime. They supply the
# domain sequence source with either a range or a value list (converting a
# value list into a range, if necessary, is the responsibility of the sequence
# source).

class DomainSurveyer(Vertex):
    def __init__(self, name, parent_space):
        super().__init__(name)
        self._space = parent_space

    @property
    def space(self):
        return self._space


class RangeCommandDomainSurveyer(DomainSurveyer):
    def __init__(self, name, parent_space, command_template):
        super().__init__(name, parent_space)
        self.command_template = command_template


class PatternDomainSurveyer(DomainSurveyer):
    pass


#
# Domain
#

# A domain represents a set of possible values along a dimension.

class Domain(Spatial):
    def __init__(self, seq_source, dimension, name=Ellipsis,
                 parent_space=None):
        self.seq_source = seq_source
        self.dimension = dimension
        self.name = name
        dimension.domains[name] = self
        self._space = (parent_space if parent_space
                       else Space())

    @property
    def space(self):
        return self._space

    @runtime
    def iterate(self, dynamic_graph, parent_point=Ellipsis):
        assert False, "abstract method call"

class FullDomain(Domain):
    @runtime
    def iterate(self, dynamic_graph, parent_point=Ellipsis):
        return iter(self.seq_source.sequence(dynamic_graph, parent_point))

class Subdomain(Domain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.space.ndims > 0:
            if not self.space.is_subspace(self.full_domain.space):
                raise ValueError("subdomain parent space must be subspace "
                                 "of corresponding full domain parent space")

    @property
    def full_domain(self):
        return self.dimension.full_domain

class SubsetDomain(Subdomain):
    @runtime
    def iterate(self, dynamic_graph, parent_point=Ellipsis):
        parent_seq = self.full_domain.seq_source.sequence(dynamic_graph,
                                                          parent_point)
        if not isinstance(parent_seq, range):
            parent_seq = frozenset(parent_seq) # Speed up.

        for value in self.seq_source.sequence(dynamic_graph, parent_point):
            if value not in parent_seq:
                raise ValueError("value generated by subset domain surveyer "
                                 "not in full domain of dimension")
            yield value

class SliceDomain(Subdomain):
    @runtime
    def iterate(self, dynamic_graph, parent_point=Ellipsis):
        slicerange = self.seq_source.sequence(dynamic_graph, parent_point)
        assert isinstance(slicerange, range)

        if len(slicerange) == 0:
            return iter(range(0))

        if len(slicerange) == 1:
            slice = slice(slicerange[0], slicerange[0] + 1)
        else:
            slice = slice(slicerange[0], slicerange[-1] + 1,
                          slicerange[1] - slicerange[0])

        parent_seq = self.full_domain.seq_source.sequence(dynamic_graph,
                                                          parent_point)
        return iter(parent_seq[slice])



#
# Space
#

# Each dataset or computation has an associated space, which is the outer
# product space formed from the set of unidimensional spaces associated with
# each dimension.

# Dimensions with parents (i.e. dimensions whose domains depend on the selected
# element in its parent dimensions) are expressed as dimensions that depend
# on the selected point in the outer product space of the unidimensional spaces
# associated with the parent dimensions.

class Space:
    def __init__(self, dimensions_domains={}):
        self.domains = OrderedDict() # Dimension -> Domain
        for dim, dom in dimensions_domains.items():
            assert dom.dimension is dim
            self.domains[dim] = dom

    @property
    def ndims(self):
        return len(self.domains)

    @runtime
    def iterate(self, dynamic_graph, parent_point=Ellipsis):
        # Yield all points of the space in order.
        if not len(self.domains):
            yield Ellipsis
            return
        dim, dom = self.domains.items()[0]

        if isinstance(dom, tuple):
            keys = tuple(dim + "." + d.name for d in dom)
            values = zip(*(d.iterate(dynamic_graph, parent_point)
                           for d in dom))
        else:
            keys = (dim,)
            values = zip(dim.iterate(dynamic_graph, parent_point))

        for coordinates in values:
            first_dim_point = list((k, v) for k, v in zip(keys, coordinates))

            rest_space = Space(self.domains.items()[1:])
            for rest_dims_point in rest_space.iterate(dynamic_graph,
                                                      first_dim_point):
                if rest_dims_point is Ellipsis:
                    yield first_dim_point
                yield first_dim_point + rest_dims_point

    def quotient_space(self, other):
        quotient_domains = OrderedDict()
        for dim in numer.domains:
            if dim in denom.domains:
                assert len(numer.domains[dim]) == len(denom.domains[dim])
                continue
            quotient_domains[dim] = numer.domains[dim]
        return self.__class__(quotient_domains)

    def is_subspace(self, other):
        # Return True if self is subspace of other.
        # Need to determine how to handle domain-domain relationships.
        return True # TODO

    def point_to_dict(self, point):
        # convert a tuple to a dict
        pass


#
# Dataset and computation
#

class Dataset(Vertex):
    def __init__(self, name, space, filename_template):
        super().__init__(name)

    def is_locally_up_to_date(self, dynamic_graph):
        # In case the dataset is not produced by a computation, we check that
        # all files are present.
        return self.mtime() > 0

class Computation(Vertex):
    def __init__(self, name, space, command_template, parallel=False):
        super().__init__(name)

    def is_locally_up_to_date(self, dynamic_graph):
        newest_input_mtime = max(input.mtime() for input
                                 in dynamic_graph.parents_of(self))
        oldest_output_ntime = min(output.mtime() for output
                                  in dynamic_graph.children_of(self))
        return newest_input_mtime < oldest_output_ntime

