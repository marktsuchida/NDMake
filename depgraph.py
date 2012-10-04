from collections import OrderedDict
import collections
import warnings
import functools
import inspect
import itertools
import shlex
import template


DEBUG = True
if DEBUG:
    def dprint(*args):
        print("depgraph: {}:".format(args[0]), *args[1:])
else:
    def dprint(*args): pass
def abstract_method_call(object_, method_name):
    class_name = type(object_).__name__
    return "abstract method {} called on {} instance".format(method_name,
                                                             class_name)


# Static data representation and dynamic execution context are partially
# separated: Instances of StaticGraph and individual model objects do not store
# any runtime state, but some methods of the model objects do act on runtime
# context. Such runtime data is stored in a DynamicGraph instance, which is
# passed around to all runtime methods. In other words, double dispatch is used
# with the model object as primary target and runtime context (dynamic_graph)
# as secondary target.

def runtime(method):
    # Decorator for dynamic method, which should have the signature
    # def f(self, dynamic_graph, ...).
    # If the runtime decorator does not implement f(self, ...), calls to f
    # will be forwarded with the dynamic_graph argument added.
    method.is_runtime_method = True
    return method


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
    def __init__(self, static_graph, runtime_decorator_factory=lambda x, y: x):
        self.static_graph = static_graph

        if isinstance(runtime_decorator_factory, collections.Mapping):
            def factory(graph, object):
                for type_, decorator_type in runtime_decorator_factory.items():
                    if type_ is Ellipsis or isinstance(object, type_):
                        return decorator_type(graph, object)
                return object
        else:
            factory = runtime_decorator_factory
        self.runtime_decorator_factory = factory

        self.runtime_decorators = {} # object -> decorator

    _forwarded_attrs = ("dimensions", "templateset",)
    def __getattr__(self, name):
        if name in _forwarded_attrs:
            return getattr(self.static_graph, name)

    def vertex_by_name(self, name, type_):
        vertex = self.static_graph.vertex_by_name(name, type_)
        return self.runtime(vertex)

    def parents_of(self, vertex):
        if isinstance(vertex, RuntimeDecorator):
            vertex = vertex.static_object
        parents = self.static_graph.parents_of(vertex)
        return list(self.runtime(v) for v in parents)

    def children_of(self, vertex):
        if isinstance(vertex, RuntimeDecorator):
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
        # Return the runtime decorator object for the given object.

        if isinstance(object, RuntimeDecorator):
            return object

        if object not in self.runtime_decorators:
            self.runtime_decorators[object] = \
                    self.runtime_decorator_factory(self, object)
        return self.runtime_decorators[object]


#
# Runtime decorator objects
#

class RuntimeDecorator:
    # We keep this simple, rather than dealing with various edge cases upfront.

    def __init__(self, dynamic_graph, static_object):
        self.graph = dynamic_graph
        self.static_object = static_object

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__,
                                str(self.static_object))

    def __getattr__(self, name):
        value = getattr(self.static_object, name)
        if inspect.ismethod(value):
            if hasattr(value.__func__, "is_runtime_method"):
                return functools.partial(value, self.graph)
        return value


#
# Vertex
#

class Vertex:
    def __init__(self, name, space):
        self.name = name
        self.space = space

    def __repr__(self):
        return "<{} \"{}\">".format(type(self).__name__, self.name)

    @property
    def namespace_type(self):
        if isinstance(self, Dataset):
            return Dataset
        if isinstance(self, Computation):
            return Computation
        if isinstance(self, DomainSurveyer):
            return DomainSurveyer

    @runtime
    def is_locally_up_to_date(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "is_locally_up_to_date")

    @runtime
    def compute(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "compute")


class VertexPlaceholder(Vertex):
    def __init__(self, name, type_):
        super().__init__(name, Space())
        self.type_ = type_

    def __repr__(self):
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
        self.domains = {} # name -> domain; "" |-> FullDomain.

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    @property
    def full_domain(self):
        return self.domains[""]


#
# Sequence sources
#

# Sequence sources supply the range or value list for a domain. Sequence
# sources can be configured with static ranges or value lists, or can use a
# domain surveyer at runtime to determine the range or value list.

class SequenceSource:
    def __init__(self, parent_space):
        self.space = parent_space

    @runtime
    def sequence(self, dynamic_graph, parent_element=None):
        assert False, abstract_method_call(self, "sequence")

class EnumeratedSequenceSource(SequenceSource):
    def __init__(self, parent_space, values):
        super().__init__(parent_space)
        self.values = values # Surveyer or template.

    @runtime
    def sequence(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()

        if isinstance(self.values, DomainSurveyer):
            # Dynamic domain.
            surveyer = dynamic_graph.runtime(self.values)
            return surveyer.values(parent_element)

        # Static domain.
        rendered_values = parent_element.render_template(dynamic_graph,
                                                         self.values)
        return shlex.split(rendered_values)

class RangeSequenceSource(SequenceSource):
    def __init__(self, parent_space, rangeargs):
        super().__init__(parent_space)
        self.rangeargs = rangeargs # Surveyer or template.

    @runtime
    def sequence(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()

        if isinstance(self.rangeargs, DomainSurveyer):
            # Dynamic domain.
            surveyer = dynamic_graph.runtime(self.rangeargs)
            values = surveyer.values(parent_element)
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
        rendered_values = parent_element.render_template(dynamic_graph,
                                                         self.values)
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
    @runtime
    def values(self, dynamic_graph, parent_element):
        assert False, abstract_method_call(self, "values")


class RangeCommandDomainSurveyer(DomainSurveyer):
    def __init__(self, name, parent_space, command_template):
        super().__init__(name, parent_space)
        self.command_template = command_template


class FilenamePatternDomainSurveyer(DomainSurveyer):
    def __init__(self, name, parent_space, match_pattern_template,
                 value_transform_template):
        super().__init__(name, parent_space)
        self.match_pattern_template = match_pattern_template
        self.value_transform_template = value_transform_template


#
# Domain
#

# A domain represents a set of possible values along a dimension.

class Domain:
    def __init__(self, seq_source, dimension, name=None,
                 parent_space=None):
        self.seq_source = seq_source
        self.dimension = dimension
        if name is None:
            assert isinstance(self, FullDomain)
            name = ""
        self.name = name
        dimension.domains[name] = self
        self.space = (parent_space if parent_space else Space())

    def __repr__(self):
        return "<{} {} of Dimension {}>".format(self.__class__.__name__,
                                                self.name,
                                                self.dimension.name)

    @runtime
    def iterate(self, dynamic_graph, parent_element=None):
        assert False, abstract_method_call(self, "iterate")

class FullDomain(Domain):
    @runtime
    def iterate(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()
        return iter(self.seq_source.sequence(dynamic_graph, parent_element))

    def __repr__(self):
        return "<{} of Dimension {}>".format(self.__class__.__name__,
                                             self.dimension.name)

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
    def iterate(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()

        parent_seq = self.full_domain.seq_source.sequence(dynamic_graph,
                                                          parent_element)
        if not isinstance(parent_seq, range):
            parent_seq = frozenset(parent_seq) # Speed up.

        for value in self.seq_source.sequence(dynamic_graph, parent_element):
            if value not in parent_seq:
                raise ValueError("value generated by subset domain surveyer "
                                 "not in full domain of dimension")
            yield value

class SliceDomain(Subdomain):
    @runtime
    def iterate(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()

        slicerange = self.seq_source.sequence(dynamic_graph, parent_element)
        assert isinstance(slicerange, range)

        if len(slicerange) == 0:
            return iter(range(0))

        if len(slicerange) == 1:
            slice = slice(slicerange[0], slicerange[0] + 1)
        else:
            slice = slice(slicerange[0], slicerange[-1] + 1,
                          slicerange[1] - slicerange[0])

        parent_seq = self.full_domain.seq_source.sequence(dynamic_graph,
                                                          parent_element)
        return iter(parent_seq[slice])



#
# Space and Element
#

# Each dataset or computation has an associated space, which is the outer
# product space formed from the set of unidimensional spaces associated with
# each dimension.

# Dimensions with parents (i.e. dimensions whose domains depend on the selected
# element in its parent dimensions) are expressed as dimensions that depend
# on the selected element in the outer product space of the unidimensional
# spaces associated with the parent dimensions.

class Space:
    def __init__(self, extent={}):
        # extent: dimension -> tuple(domains)
        self.extent = OrderedDict() # Dimension -> Domain
        for dim, doms in extent.items():
            for dom in doms:
                assert dom.dimension is dim
            self.extent[dim] = doms

        # XXX make a topologically sorted dims list!

    def __repr__(self):
        dims = ("{}.{}".
                format(dim.name, "|".join(dom.name for dom in doms)).
                rstrip(".")
                for dim, doms in self.extent.items())
        return "<Space [{}]>".format(", ".join(dims))

    @property
    def ndims(self):
        return len(self.extent)

    @property
    def dimensions(self):
        return self.extent.keys()

    @runtime
    def iterate(self, dynamic_graph, parent_element=None):
        if parent_element is None:
            parent_element = Element()

        # Sanity check: TODO Remove.
        for dim in parent_element.space.dimensions:
            assert dim not in self.dimensions

        # Yield all elements of the space in order.
        if not len(self.extent):
            if parent_element.space.ndims > 0:
                yield parent_element
            else:
                yield Element()
            return

        # The dimension to iterate along: the first dimension.
        extent_items = list(self.extent.items())
        dimension, domains = extent_items[0]

        # Extend the parent space with the first dimension.
        new_parent_space = (parent_element.space.
                            space_by_adding_dimension(dimension, domains))
        for coord in zip(*(dom.iterate(dynamic_graph, parent_element)
                           for dom in domains)):
            coords = parent_element.coordinates.copy()
            coords[dimension] = coord
            element = Element(new_parent_space, coords)

            remaining_space = Space(OrderedDict(extent_items[1:]))
            for full_element in remaining_space.iterate(dynamic_graph,
                                                        element):
                yield full_element

    def space_by_adding_dimension(self, dimension, domains):
        extent = self.extent.copy()
        extent[dimension] = domains
        return Space(extent)

    def quotient_space(self, other):
        quotient_domains = OrderedDict()
        for dim in numer.extent:
            if dim in denom.extent:
                assert len(numer.extent[dim]) == len(denom.extent[dim])
                continue
            quotient_domains[dim] = numer.extent[dim]
        return self.__class__(quotient_domains)

    def is_compatible_space(self, other):
        warnings.warn("TODO: Space.is_compatible_space")
        return True # TODO

    def is_subspace(self, other):
        # Return True if self is subspace of other.
        # Need to determine how to handle domain-domain relationships.
        warnings.warn("TODO: Space.is_subspace")
        return True # TODO


class Element:
    # An immutable "vector" (or "point") in a Space.
    def __init__(self, space=Space(), coordinates=dict()):
        self.space = space
        # XXX
        self.coordinates = dict((dim, coordinates[dim])
                                for dim in self.space.dimensions)

    def __hash__(self):
        # XXX
        return hash(tuple(self.coordinates[dim]
                          for dim in self.space.dimensions))

    def __eq__(self, other):
        # XXX
        self_tuple = tuple(self.coordinates[dim]
                           for dim in self.space.dimensions)
        other_tuple = tuple(other.coordinates[dim]
                            for dim in other.space.dimensions)
        return self_tuple == other_tuple

    def __repr__(self):
        # XXX WRONG need parent domains
        dims = ("{}.{}".
                format(dim.name, "|".join(dom.name for dom in doms)).
                rstrip(".")
                for dim, doms in self.space.extent.items())
        coords = (self.coordinates[dim] for dim in self.space.dimensions)
        return "<Element [{}]>".format(", ".join("{} = {}".format(dim, coord)
                                               for dim, coord
                                               in zip(dims, coords)))

    def as_dict(self):
        # XXX
        dict_ = {}
        for dimension in self.space.dimensions:
            for i, domain in enumerate(self.space.extent[dimension]):
                if isinstance(domain, FullDomain):
                    key = dimension.name
                else:
                    key = "{}.{}".format(dimension.name, domain.name)
                dict_[key] = self.coordinates[dimension][i]
        return dict_

    @runtime
    def filename_expander(self, dynamic_graph):
        def expander(dataset_name, extra_coords):
            dataset = dynamic_graph.vertex_by_name(dataset_name, Dataset)
            return self.render_template(dynamic_graph,
                                        dataset.filename_template,
                                        extra_names=extra_coords)
        return expander

    @runtime
    def render_template(self, dynamic_graph, template,
                        expand_datasets=False, extra_names={}):
        dict_ = self.as_dict().copy()
        dict_.update(extra_names)
        expander = (self.filename_expander(dynamic_graph)
                    if expand_datasets else None)
        rendition = template.render(dict_, expander)
        return rendition


#
# Dataset and computation
#

class Dataset(Vertex):
    def __init__(self, name, space, filename_template):
        super().__init__(name, space)
        self.filename_template = filename_template

class Computation(Vertex):
    def __init__(self, name, space, command_template, parallel=False):
        super().__init__(name, space)
        self.command_template = command_template
        self.parallel = parallel


