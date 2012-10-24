import collections
import functools
import inspect
import itertools
import shlex

from ndmake import debug
from ndmake import template


dprint = debug.dprint_factory(__name__)
dprint_iter = debug.dprint_factory(__name__, "iter")
dprint_extent = debug.dprint_factory(__name__, "extent")

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

    def write_graphviz(self, filename):
        with open(filename, "w") as file:
            fprint = functools.partial(print, file=file)
            fprint("digraph depgraph {")
            for id, vertex in self._id_vertex_map.items():
                label = vertex.name
                shape = "box"
                color = "black"
                if isinstance(vertex, Computation):
                    shape = "box"
                    color = "black"
                elif isinstance(vertex, Dataset):
                    shape = "folder"
                    color = "navy"
                elif isinstance(vertex, Survey):
                    label = " ".join((vertex.__class__.__name__, vertex.name))
                    shape = "box"
                    color = "red"
                fprint("v{:d} [label=\"{}\" shape=\"{}\" color=\"{}\"];".
                       format(id, label, shape, color))
            for parent, children in self._parents.items():
                for child in children:
                    fprint("v{:d} -> v{:d};".format(child, parent))
            fprint("}")

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
    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    def __repr__(self):
        return "<{} \"{}\">".format(type(self).__name__, self.name)

    @property
    def namespace_type(self):
        if isinstance(self, Dataset):
            return Dataset
        if isinstance(self, Computation):
            return Computation
        if isinstance(self, Survey):
            return Survey


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
# Concrete Vertices
#

class Dataset(Vertex):
    def __init__(self, name, scope, filename_template):
        super().__init__(name, scope)
        self.filename_template = filename_template


class Computation(Vertex):
    def __init__(self, name, scope, command_template, occupancy=1):
        super().__init__(name, scope)
        self.command_template = command_template
        self.occupancy = occupancy


class Survey(Vertex):
    def __init__(self, name, scope, surveyer):
        super().__init__(name, scope)
        self.surveyer = surveyer


#
# Surveyer
#

class Surveyer:
    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    @runtime
    def survey(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "survey")

class CommandSurveyer(Surveyer):
    def __init__(self, name, scope, command_template):
        super().__init__(name, scope)
        self.command_template = command_template

class IntegerTripletCommandSurveyer(CommandSurveyer):
    pass

class ValuesCommandSurveyer(CommandSurveyer):
    def __init__(self, name, scope, command_template, transform_template=None):
        super().__init__(name, scope)

class FilenameSurveyer(Surveyer):
    def __init__(self, name, scope, pattern_template, transform_template=None):
        super().__init__(name, scope)

#
# Dimension
#

class Dimension:
    def __init__(self, name):
        self.name = name
        self.full_extent = None

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def extent_by_name(self, name):
        names = name.split(".")
        assert names[0] == self.name
        extent = self.full_extent
        for name in names[1:]:
            extent = extent.subextents[name]
        return extent


#
# Extent
#

# An extent encapsulates the sequence of possible values for a dimension.

class Extent:
    def __init__(self, source):
        self.source = source # Template or Survey
        self.name = None
        self.superextent = None
        self.subextents = {} # name -> Extent

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.full_name)

    @property
    def full_name(self):
        assert False, abstract_method_call(self, "full_name")

    @property
    def is_surveyed(self):
        return isinstance(self.source, Survey)

    @runtime
    def is_demarcated(self, dynamic_graph, element):
        if not self.is_surveyed:
            return True
        survey = dynamic_graph.runtime(self.source)
        return not not survey.is_result_available(element)

    @runtime
    def sequence(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "sequence")

    @runtime
    def iterate(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "iterate")

    def issubextent(self, other):
        if self is other:
            return True
        elif self.superextent is not None:
            if self.superextent.issubextent(other):
                return True
        return False


class SequenceExtentMixin:
    def __init__(self, *args, **kwargs): pass

    @runtime
    def raw_sequence(self, dynamic_graph, element):
        assert False, abstract_method_call(self, "raw_sequence")


class EnumeratedExtentMixin(SequenceExtentMixin):
    @runtime
    def raw_sequence(self, dynamic_graph, element):
        if isinstance(self.source, Survey):
            survey = dynamic_graph.runtime(self.source)
            return survey.result(element)

        values_str = element.render_template(self.source)
        dprint_extent(self, "values string:", values_str)
        values = shlex.split(values_str)
        dprint_extent(self, "values:", ", ".join(values))
        return values


class ArithmeticExtentMixin(SequenceExtentMixin):
    @runtime
    def raw_sequence(self, dynamic_graph, element):
        if isinstance(self.source, Survey):
            survey = dynamic_graph.runtime(self.source)
            rangeargs = survey.result(element)

        else:
            rangeargs_str = element.render_template(self.source)
            dprint_extent(self, "rangeargs string:", rangeargs_str)
            try:
                rangeargs = tuple(int(a) for a in rangeargs_str.split())
                assert len(rangeargs) in range(1, 4)
            except:
                raise ValueError("{}: range template must expand to 1-3 "
                                 "integers (got `{}')".
                                 format(self.full_name, rangeargs_str))

        dprint_extent(self, "rangeargs:", ", ".join(str(a) for a in rangeargs))
        values = list(str(i) for i in range(*rangeargs))
        return values


class FullExtent(Extent, SequenceExtentMixin):
    def __init__(self, dimension, scope, source):
        super().__init__(source)
        self.dimension = dimension
        self.scope = scope
        # TODO Check that self.dimension is not in self.scope.extents

    @property
    def full_name(self):
        return self.dimension.name

    @runtime
    def sequence(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        # raw_sequence() implemented by subclasses.
        return self_runtime.raw_sequence(element)

    @runtime
    def iterate(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        return iter(self_runtime.sequence(element))


class EnumeratedFullExtent(FullExtent, EnumeratedExtentMixin):
    pass


class ArithmeticFullExtent(FullExtent, ArithmeticExtentMixin):
    pass


class Subextent(Extent):
    def __init__(self, superextent, name, source):
        super().__init__(source)
        self.name = name
        self.superextent = superextent
        self.dimension = superextent.dimension
        self.scope = superextent.scope

    @property
    def full_name(self):
        return self.superextent.full_name + "." + self.name


class SequenceSubextent(Subextent, SequenceExtentMixin):
    @runtime
    def sequence(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        return list(self_runtime.iterate(element))

    @runtime
    def iterate(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        super_runtime = dynamic_graph.runtime(self.superextent)
        super_values = frozenset(super_runtime.sequence(element))
        for value in self_runtime.raw_sequence(element):
            if value not in super_values:
                super_values = " ".join(super_runtime.sequence(element))
                if not super_values:
                    super_values = "(none)"
                raise ValueError("value `{value}' generated by {sub} "
                                 "not a member of {super} "
                                 "(allowed values are: {supervalues})".
                                 format(sub=self.full_name,
                                        super=self.superextent.full_name,
                                        value=value,
                                        supervalues=super_values))
            yield value


class EnumeratedSubextent(SequenceSubextent, EnumeratedExtentMixin):
    pass


class ArithmeticSubextent(SequenceSubextent, ArithmeticExtentMixin):
    pass


class IndexedSubextent(Subextent):
    @runtime
    def slice(self, dynamic_graph, element):
        if isinstance(self.source, Survey):
            survey = dynamic_graph.runtime(self.source)
            sliceargs = survey.result(element)

        else:
            sliceargs_str = element.render_template(self.source)
            dprint_extent(self, "sliceargs string:", sliceargs_str)
            try:
                sliceargs = tuple(int(a) for a in sliceargs_str.split())
                assert len(sliceargs) in range(1, 4)
            except:
                raise ValueError("{}: slice template must expand to 1-3 "
                                 "integers (got `{}')".
                                 format(self.full_name, sliceargs_str))

        dprint_extent(self, "sliceargs:", ", ".join(str(a) for a in sliceargs))
        return slice(*sliceargs)

    @runtime
    def sequence(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        super_runtime = dynamic_graph.runtime(self.superextent)
        return super_runtime.sequence(element)[self_runtime.slice]

    @runtime
    def iterate(self, dynamic_graph, element):
        self_runtime = dynamic_graph.runtime(self)
        super_runtime = dynamic_graph.runtime(self.superextent)
        slice_ = self_runtime.slice(element)
        sliceargs = (slice_.start, slice_.stop, slice_.step)
        return iter(list(super_runtime.iterate(element))[slice(*sliceargs)])


#
# Space and Element
#

class Space:
    # The __init__() implemented below can take up a significant portion of the
    # time taken to iterate over spaces, so we memoize Space instances.
    # There are more general and sophisticated methods to memoize class
    # instances, but for now we assume that Space will not be subclassed.
    _instances = {} # tuple(manifest_extents) -> Space
    def __new__(cls, manifest_extents=[]):
        key = tuple(manifest_extents)
        if key not in cls._instances:
            cls._instances[key] = super(Space, cls).__new__(cls)
        return cls._instances[key]

    def __init__(self, manifest_extents=[]):
        if hasattr(self, "manifest_extents"):
            # Cached instance.
            return

        manifest_dims = frozenset(extent.dimension
                                  for extent in manifest_extents)
        if len(manifest_extents) > len(manifest_dims):
            raise ValueError("cannot create space with nonorthogonal "
                             "extents: {}".
                             format(", ".join(str(e)
                                              for e in manifest_extents)))
        self.manifest_extents = manifest_extents

        # Collect all the extents on which the manifest ones depend on.
        # If extents for the same dimension are encountered (between the scopes
        # of different manifest contingent extents or between the scope of
        # a manifest contingent extent and one of the manifest extents), then
        # they must be in a superextent-subextent relationship, and the
        # subextent is adopted as the extent for the whole space.
        extents_to_use = dict((extent.dimension, extent)
                              for extent in self.manifest_extents)
        for extent in self.manifest_extents:
            for extent in extent.scope.extents:
                dimension = extent.dimension
                if dimension in extents_to_use:
                    contending_extent = extents_to_use[dimension]
                    if extent.issubextent(contending_extent):
                        extents_to_use[dimension] = extent
                        continue
                    elif contending_extent.issubextent(extent):
                        continue # Keep contending_extent for this dimension.
                    raise ValueError("cannot create space with multiple "
                                     "incompatible extents for the same "
                                     "dimension: {}".format(manifest_extents))
                else:
                    extents_to_use[dimension] = extent

        # Topologically sort the extents based on contingency relations.

        # In order to preserve the order of the manifest extents (so that the
        # user can specify the order of iteration), for each manifest extent M
        # (in order), we "claim" any remaining implicit extents {I} on which M
        # is contingent and place them to the left of M (preserving the
        # already topologically-sorted order of {I}). At the same time, we
        # make sure that the manifest extents are in fact topologically sorted.
        all_dims = set(extents_to_use.keys())
        implicit_dims = all_dims - manifest_dims
        sorted_extents = []
        checked_dimensions = set() # Dimensions represented in sorted_extents.
        for manifest_extent in self.manifest_extents:
            if manifest_extent.dimension in checked_dimensions:
                raise ValueError("cannot create space with extents that are "
                                 "not topologically sorted")
            for e in manifest_extent.scope.extents:
                d = e.dimension
                if d in implicit_dims:
                    implicit_dims.remove(d)
                    sorted_extents.append(extents_to_use[d])
                    checked_dimensions.add(d)
            sorted_extents.append(extents_to_use[manifest_extent.dimension])
            checked_dimensions.add(manifest_extent.dimension)
        
        self.extents = sorted_extents
        # Some useful attributes:
        self.dimensions = list(e.dimension for e in self.extents)
        self.ndims = len(self.dimensions)

    def __repr__(self):
        return "<Space [{}]>".format(", ".join(e.full_name
                                               for e in self.extents))

    def __getitem__(self, dimension):
        for extent in self.extents:
            if extent.dimension is dimension:
                return extent
        raise KeyError("{} not in {}".format(dimension, self))

    def is_full_element(self, element):
        for extent in self.extents:
            if extent.dimension not in element.space.dimensions:
                return False
        return True

    def canonicalized_element(self, element, require_full=False):
        if element.space is self:
            return element

        assigned_extents = []
        coords = {}
        element_space_dimensions = element.space.dimensions
        for extent in self.extents:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                if require_full:
                    raise ValueError("{} is not an element of {}".
                                     format(element, self))
                continue
            if not element.space[dimension].issubextent(extent):
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            assigned_extents.append(extent)
            coords[dimension] = element[dimension]
        return Element(Space(assigned_extents), coords)

    @runtime
    def iterate(self, dynamic_graph, element=None):
        if element is None:
            element = Element()

        # Scan _consecutive_ dimensions assigned a value by element.
        assigned_extents = []
        base_coords = {}
        element_space_dimensions = element.space.dimensions
        for extent in self.extents:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                break
            if not element.space[dimension].issubextent(extent):
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            assigned_extents.append(extent)
            base_coords[dimension] = element[dimension]

        if len(assigned_extents) == self.ndims:
            # Element assigned coordinates to all of our dimensions.
            # (We generate a canonical element directly, without having to call
            # self.canonicalized_element().)
            element = Element(self, base_coords)
            dprint_iter(self, "iterate({}): yielding full element".
                        format(element))
            yield (element, True) # Flag indicating full element.
            return

        # We will iterate over the first unassigned extent.
        extent_to_iterate = self.extents[len(assigned_extents)]
        assigned_extents.append(extent_to_iterate)

        # Scan remaining (nonconsecutive) dimensions assigned by element.
        for extent in self.extents[len(assigned_extents):]:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                continue
            if not element.space[dimension].issubextent(extent):
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            assigned_extents.append(extent)
            base_coords[dimension] = element[dimension]

        extent_runtime = dynamic_graph.runtime(extent_to_iterate)
        if extent_runtime.is_demarcated(element):
            dprint_iter(self, "iterate({}): iterating extent:".format(element),
                        extent_to_iterate)
            for value in (extent_runtime.iterate(element)):
                base_coords[extent_to_iterate.dimension] = value
                new_element = Element(Space(assigned_extents), base_coords)
                dprint_iter(self, "iterate({}): descending into:".
                            format(element), new_element)
                yield from dynamic_graph.runtime(self).iterate(new_element)
            dprint_iter(self, "iterate({}): finished".format(element))
        else: # Undemarcated: yield a partial element.
            assigned_extents.remove(extent_to_iterate)
            new_element = Element(Space(assigned_extents), base_coords)
            dprint_iter(self, "iterate({}) yielding partial element:".
                        format(element), new_element)
            yield (new_element, False) # Flag indicating partial element.



class Element:
    def __init__(self, space=Space(), coordinates=dict()):
        self.space = space
        self.coordinates = {}
        for extent in self.space.extents:
            dim = extent.dimension
            if dim in coordinates:
                self.coordinates[dim] = coordinates[dim]
                continue
            assert False, ("attempt to construct incomplete Element for {}: "
                           "coordinates: {}".format(space, coordinates))

    def __getitem__(self, dimension):
        return self.coordinates[dimension]

    def __hash__(self):
        return hash(tuple(self.coordinates[dim]
                          for dim in self.space.dimensions))

    def __eq__(self, other):
        self_tuple = tuple(self.coordinates[dim]
                           for dim in self.space.dimensions)
        other_tuple = tuple(other.coordinates[dim]
                            for dim in other.space.dimensions)
        return self_tuple == other_tuple

    def __repr__(self):
        coords = ("{}={}".format(extent.full_name, self[extent.dimension])
                  for extent in self.space.extents)
        return "<Element [{}]>".format(", ".join(coords))

    def as_dict(self):
        return dict((dim.name, self[dim]) for dim in self.space.dimensions)

    def render_template(self, template, *, extra_names={}):
        dict_ = self.as_dict().copy()
        dict_.update(extra_names)
        rendition = template.render(dict_)
        return rendition

