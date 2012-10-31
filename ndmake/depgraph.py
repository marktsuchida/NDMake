import functools
import itertools
import multiprocessing
import os.path
import shlex
import subprocess
import sys
import time

from ndmake import debug
from ndmake import dispatch
from ndmake import files
from ndmake import mux
from ndmake import template
from ndmake import threadpool


dprint = debug.dprint_factory(__name__)
dprint_iter = debug.dprint_factory(__name__, "iter")
dprint_extent = debug.dprint_factory(__name__, "extent")
dprint_traverse = debug.dprint_factory(__name__, "traverse")
dprint_update = debug.dprint_factory(__name__, "update")
_dprint_mtime = debug.dprint_factory(__name__, "mtime")

def strfmtime(mtime):
    return (time.strftime("%Y%m%dT%H%M%S", time.localtime(mtime)) +
            ".{:04d}".format(round(mtime % 1 * 10000)))

def dprint_mtime(mtime, *args):
    # Wrapper to avoid strftime() call when not printing.
    if debug.dprint_enabled.get(__name__ + "_mtime"):
        if isinstance(mtime, int) or isinstance(mtime, float):
            _dprint_mtime("mtime", strfmtime(mtime), *args)
        else:
            _dprint_mtime("mtime", mtime, *args)


# A constant representing a "valid" time-since-epoch value.
# This is only done to prevent errors when printing ctime(MAX_TIME).
MAX_TIME = 2**64 - 1
while True:
    try:
        time.asctime(time.gmtime(MAX_TIME))
        time.ctime(MAX_TIME)
    except:
        MAX_TIME >>= 1
    else:
        # Clip to a recognizable maximum to aid debugging.
        max_max_time = time.mktime((2999, 12, 31, 23, 59, 59, 0, 0, 0))
        MAX_TIME = min(MAX_TIME, max_max_time)
        break


#
# Exceptions
#

class UpdateException(Exception): pass
class MissingFileException(UpdateException): pass
class CalledProcessError(UpdateException): pass


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

    def iterate(self, element=None):
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

        if extent_to_iterate.is_demarcated(element):
            dprint_iter(self, "iterate({}): iterating extent:".format(element),
                        extent_to_iterate)
            for value in (extent_to_iterate.iterate(element)):
                base_coords[extent_to_iterate.dimension] = value
                new_element = Element(Space(assigned_extents), base_coords)
                dprint_iter(self, "iterate({}): descending into:".
                            format(element), new_element)
                yield from self.iterate(new_element)
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


#
# Graph façade
#

class Graph:
    # Vertices and edges are managed by integer vertex ids. The Vertex objects
    # themselves do not hold pointers to each other.
    #
    # The acyclicity of the directed graph is enforced during graph
    # construction. XXX Might be more efficient to check later.

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

    @dispatch.tasklet
    def update_vertices(self, vertices, **options):
        for vertex in vertices:
            yield dispatch.Spawn(vertex.update(self, **options))

        # Wait for completion.
        notification_chans = [] # Can't use generator expression here.
        for vertex in vertices:
            notification_chan = yield from vertex.get_notification_chan()
            notification_chans.append(notification_chan)
        if len(notification_chans):
            completion_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(mux.demultiplex(notification_chans),
                                 return_chan=completion_chan)
            yield dispatch.Recv(completion_chan)

    @dispatch.tasklet
    def update_vertices_with_threadpool(self, vertices, **options):
        new_options = options.copy()
        if options.get("parallel", False):
            jobs = options.get("jobs")
            if not jobs or jobs < 1:
                jobs = multiprocessing.cpu_count()
            task_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(threadpool.threadpool(task_chan, jobs))
            new_options["threadpool"] = task_chan

        yield from self.update_vertices(vertices, **new_options)

        if "threadpool" in new_options:
            finish_chan = yield dispatch.MakeChannel()
            yield dispatch.Send(task_chan, (..., None, finish_chan, None),
                                block=False)
            yield dispatch.Recv(finish_chan)


#
# Vertex
#

class Vertex:
    def __init__(self, graph, name, scope):
        self.name = name
        self.scope = scope

        self.update_started = False # Prevent duplicate update.
        self.notification_request_chan = None # Request chan for mux.

        self.mtimes = ElementMTimeCache(self.scope, self.read_mtimes)

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

    def newest_mtime(self, element=Element()):
        element = self.scope.canonicalized_element(element)
        _, newest = self.mtimes[element]
        return newest

    def oldest_mtime(self, element=Element()):
        element = self.scope.canonicalized_element(element)
        oldest, _ = self.mtimes[element]
        return oldest

    @dispatch.tasklet
    def update_all_elements(self, graph, **options):
        dprint_update(self, "updating all elements")
        completion_chans = []
        for element, is_full in self.scope.iterate():
            completion_chan = yield dispatch.MakeChannel()
            completion_chans.append(completion_chan)
            yield dispatch.Spawn(self.update_element(graph, element, is_full,
                                                     **options),
                                 return_chan=completion_chan)

            # With the current implementation of dispatch.py, having a large
            # number of channels with pending messages slows down the scheduler
            # significantly. Until this issue is fixed (if ever), we keep down
            # the number of active channels by preemptively demultiplexing the
            # completion notification channels. The chunk size of 11 has been
            # determined empirically, but run time is roughly constant with
            # chunk sizes of 2-32.
            if len(completion_chans) > 11:
                chunk_complete_chan = yield dispatch.MakeChannel()
                yield dispatch.Spawn(mux.demultiplex(completion_chans),
                                     return_chan = chunk_complete_chan)
                completion_chans = [chunk_complete_chan]

        all_complete_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(mux.demultiplex(completion_chans),
                             return_chan=all_complete_chan)
        yield dispatch.Recv(all_complete_chan)

    @dispatch.subtasklet
    def get_notification_request_chan(self):
        # Return the unique channel attached to each vertex to which
        # notification requests can be sent. Notification is requested by
        # sending the handle to the notification channel to the channel
        # returned by this subtasklet. All notification channels receive a
        # signal exactly once, after the vertex has been updated or as soon as
        # the request is made, whichever comes later.
        if self.notification_request_chan is None:
            self.notification_request_chan = yield dispatch.MakeChannel()
        return self.notification_request_chan

    @dispatch.subtasklet
    def get_notification_chan(self):
        # Return a new channel that will receive notification of completion
        # of update of this vertex. The channel will recieve a signal even if
        # update has already been completed by the time this subtasklet is
        # called.
        request_chan = (yield from self.get_notification_request_chan())

        # Create and register a new notification channel.
        notification_chan = yield dispatch.MakeChannel()
        yield dispatch.Send(request_chan, notification_chan, False)

        return notification_chan

    @dispatch.tasklet
    def update(self, graph, **options):
        # Prevent duplicate execution.
        if self.update_started:
            return
        self.update_started = True

        # Set up a channel by which completion notification channels can be
        # registered.
        request_chan = yield from self.get_notification_request_chan()

        # Set up notification for our completion.
        completion_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(mux.multiplex(completion_chan, request_chan))
        yield dispatch.Spawn(self._update(graph, **options),
                             return_chan=completion_chan)

    @dispatch.tasklet
    def _update(self, graph, **options):
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing upward:", self)

        # Update prerequisites.
        parents = graph.parents_of(self)
        yield from graph.update_vertices(parents, **options)

        # Perform the update action.
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing downward:", self)
        print("starting check/update of {}".format(self))
        completion_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(self.update_all_elements(graph, **options),
                             return_chan=completion_chan)
        yield dispatch.Recv(completion_chan)
        print("finished check/update of {}".format(self))


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
    def __init__(self, graph, name, scope, filename_template):
        super().__init__(graph, name, scope)
        self.filename_template = filename_template

    def read_mtimes(self, element):
        filename = element.render_template(self.filename_template)
        try:
            mtime = os.path.getmtime(filename)
            dprint_mtime(mtime, filename)
        except FileNotFoundError:
            dprint_mtime("missing", filename)
            return 0, MAX_TIME
        return mtime, mtime # oldest, newest

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, **options):
        if 0: yield
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        oldest_mtime, newest_mtime = self.mtimes[element]
        # Since the dataset will not further change, we can cache the mtimes.
        self.mtimes[element] = (oldest_mtime, newest_mtime)

        if ... in (oldest_mtime, newest_mtime): # Partial element.
            assert not is_full
            return
        assert is_full

        if oldest_mtime == 0 or newest_mtime == MAX_TIME:
            # There were missing files.
            # Unless this is a dry run or a keep-going run, we raise an error.
            # XXX For now, we raise an error unconditionally.
            filename = element.render_template(self.filename_template)
            parents = graph.parents_of(self)
            for parent in parents:
                if isinstance(parent.static_object, Computation):
                    raise MissingFileException("file {filename} (member of "
                                               "dataset {dataset}; output of "
                                               "compute {compute}) missing".
                                               format(filename=filename,
                                                      dataset=self.name,
                                                      compute=parent.name))
            raise MissingFileException("file {filename} (member of source "
                                       "dataset {dataset}) missing".
                                       format(filename=filename,
                                              dataset=self.name))

            # In the future, there should be a way to associate with this
            # exception the command that should have produced the file (if
            # not a source dataset).

    def dirname(self, element):
        element = self.scope.canonicalized_element(element)
        if self.scope.is_full_element(element):
            filename = element.render_template(self.filename_template)
            return os.path.dirname(filename)
        else:
            # Given a partial element, we still want to be able to create
            # directories whose names are fixed. There might be a better way
            # to do this, but for now, we set the parameters in the
            # undemarcated subspace to a likely-unique marker string and remove
            # path components from the end until the marker string no longer
            # appears in the result.
            unassigned_marker = "@@@@@UNASSIGNED@@@@@"
            coords = {}
            for extent in self.scope.extents:
                if extent.dimension in element.space.dimensions:
                    coords[extent.dimension] = element[extent.dimension]
                    continue
                coords[extent.dimension] = unassigned_marker
            full_element = Element(self.scope, coords)
            filename = full_element.render_template(self.filename_template)
            dirname = os.path.dirname(filename)
            while unassigned_marker in dirname:
                dirname = os.path.dirname(dirname)
                if not dirname:
                    return None
            return dirname

    def name_proxy(self, element):
        return DatasetNameProxy(self, element)


class Computation(Vertex):
    def __init__(self, graph, name, scope, command_template, occupancy=1):
        super().__init__(graph, name, scope)
        self.command_template = command_template
        self.occupancy = occupancy

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, **options):
        if 0: yield
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        if not is_full:
            # XXX Print if requested.
            self.mtimes[element] = (..., ...)
            return

        parents = graph.parents_of(self)
        newest_input_mtime = 0
        for parent in parents:
            assert isinstance(parent, Dataset)
            parent_newest_mtime = parent.newest_mtime(element)
            newest_input_mtime = max(newest_input_mtime, parent_newest_mtime)

        previous_starttime, previous_finishtime = self.read_mtimes(element)

        if previous_finishtime < newest_input_mtime:
            # In this case we know we are out of date and need not check the
            # dataset files.
            pass
        else:
            children = graph.children_of(self)
            oldest_output_mtime = MAX_TIME
            for child in children:
                oldest_output_mtime = min(oldest_output_mtime,
                                          child.oldest_mtime(element))
            
            if newest_input_mtime <= oldest_output_mtime:
                # We are up to date.
                self.mtimes[element] = (previous_starttime,
                                        previous_finishtime)
                return

        # We are not up to date, so run the computation.
        self.create_dirs_for_output(graph, element)
        self.touch_starttime_stamp(element)
        yield from self.run_command(graph, element, **options)
        self.touch_finishtime_stamp(element)

    def read_mtimes(self, element):
        startstamp = os.path.join(self.cache_dir(element), "start")
        try:
            start = os.path.getmtime(startstamp)
            dprint_mtime(start, startstamp)
        except FileNotFoundError:
            dprint_mtime("missing", startstamp)
            start = 0

        finishstamp = os.path.join(self.cache_dir(element), "finish")
        try:
            finish = os.path.getmtime(finishstamp)
            dprint_mtime(finish, finishstamp)
        except FileNotFoundError:
            dprint_mtime("missing", finishstamp)
            finish = MAX_TIME

        return start, finish

    def create_dirs_for_output(self, graph, element, **options):
        for child in graph.children_of(self):
            dirnames = set()
            dirname = child.dirname(element)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

    def cache_dir(self, element):
        return os.path.join(files.ndmake_dir(), "compute", self.name,
                            files.element_dirs(element))

    def touch_starttime_stamp(self, element):
        stamp = os.path.join(self.cache_dir(element), "start")
        mtime = files.touch(stamp)
        self.mtimes[element] = (mtime, MAX_TIME)

    def touch_finishtime_stamp(self, element):
        stamp = os.path.join(self.cache_dir(element), "finish")
        mtime = files.touch(stamp)
        starttime, _ = self.mtimes[element]
        self.mtimes[element] = (starttime, mtime)

    @dispatch.subtasklet
    def run_command(self, graph, element, **options):
        if 0: yield

        # Bind input and output dataset names.
        io_vertices = (graph.parents_of(self) + graph.children_of(self))
        dataset_name_proxies = dict((v.name, v.name_proxy(element))
                                    for v in io_vertices
                                    if hasattr(v, "name_proxy"))
        # Note: filename-surveyed output datasets' names are not available in
        # the command template.

        command = element.render_template(self.command_template,
                                          extra_names=dataset_name_proxies)

        def task_func():
            # Avoid print(), which apparently flushes the output between the
            # string and the newline.
            sys.stdout.write(command + "\n")

            with subprocess.Popen(command, shell=True) as proc:
                # TODO Capture stdout and stderr (need select.select())
                proc.wait()
                return proc.returncode

        if "threadpool" in options:
            completion_chan = yield dispatch.MakeChannel()
            yield dispatch.Send(options["threadpool"],
                                (task_func, None, completion_chan,
                                 self.occupancy))
            # Wait for task_func() to finish.
            _, retval, exc_info = yield dispatch.Recv(completion_chan)
            if exc_info:
                raise exc_info
        else:
            # Run serially.
            retval = task_func()

        if retval:
            # TODO Remove output files (by calling dsets and match surveys)
            # For now, we raise unconditionally.
            raise CalledProcessError("command returned exit status of "
                                     "{:d}: {}".format(retval, command))


class Survey(Vertex):
    def __init__(self, graph, name, scope, surveyer):
        super().__init__(graph, name, scope)
        self.surveyer = surveyer

        self.results = {}

    def is_result_available(self, element):
        # Iff we've been updated, the results are stored in self.results.
        element = self.scope.canonicalized_element(element)
        return element in self.results

    def result(self, element):
        element = self.scope.canonicalized_element(element)
        return self.results[element]

    def read_mtimes(self, element):
        return self.surveyer.read_mtimes(element)

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, **options):
        if 0: yield
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        if not is_full:
            self.results[element] = ... # Not really necessary.
            return

        parents = graph.parents_of(self)
        newest_input_mtime = 0
        for parent in parents:
            assert isinstance(parent, Dataset)
            newest_input_mtime = max(newest_input_mtime,
                                     parent.newest_mtime(element))

        previous_oldest, previous_newest = self.surveyer.read_mtimes(element)
        if newest_input_mtime <= previous_oldest:
            # We are up to date; load previous results.
            self.results[element] = self.surveyer.read_result(element)
            self.mtimes[element] = (previous_oldest, previous_newest)
            return

        # We are not up to date, so do the survey.
        self.results[element] = self.surveyer.survey(graph, self, element)
        self.mtimes[element] = self.surveyer.read_mtimes(element)


#
# Surveyer
#

class Surveyer:
    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    def read_mtimes(self, element):
        filename = self.cache_file(element)
        try:
            mtime = os.path.getmtime(filename)
            dprint_mtime(mtime, filename)
        except FileNotFoundError:
            dprint_mtime("missing", filename)
            return 0, MAX_TIME
        return mtime, mtime # oldest, newest

    def read_result(self, element):
        with open(self.cache_file(element)) as file:
            result_text = file.read()
        return result_text

class CommandSurveyer(Surveyer):
    def __init__(self, name, scope, command_template):
        super().__init__(name, scope)
        self.command_template = command_template

    def cache_file(self, element):
        return os.path.join(files.ndmake_dir(), "survey", self.name,
                            files.element_dirs(element), "command_output")

    def read_result(self, element):
        result_text = super().read_result(element)
        return self.convert_result(result_text)

    def survey(self, graph, survey, element):
        # Bind input dataset names.
        parents = graph.parents_of(survey)
        dataset_name_proxies = dict((parent.name, parent.name_proxy(element))
                                    for parent in parents
                                    if isinstance(parent, Dataset))

        command = element.render_template(self.command_template,
                                          extra_names=dataset_name_proxies)

        print(command)
        # TODO Command should probably run in thread.
        with subprocess.Popen(command, shell=True,
                              bufsize=-1, universal_newlines=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as proc:
            result_text, error_text = proc.communicate()
            if error_text:
                print(error_text, file=sys.stderr)
            if proc.returncode:
                raise CalledProcessError("command returned exit status of "
                                         "{:d}: {}".format(proc.returncode,
                                                           command))

        result = self.convert_result(result_text)

        cache_file = self.cache_file(element)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as file:
            file.write(result_text)

        return result


class IntegerTripletCommandSurveyer(CommandSurveyer):
    def convert_result(self, result_text):
        try:
            result = tuple(int(v) for v in result_text.split())
            assert len(result) in range(1, 4)
        except:
            raise ValueError("command output does not conform to range or "
                             "slice argument format (1-3 integers required)")
        return result


class ValuesCommandSurveyer(CommandSurveyer):
    def __init__(self, name, scope, command_template, transform_template=None):
        super().__init__(name, scope)

    def convert_result(self, result_text):
        return result_text.splitlines()


class FilenameSurveyer(Surveyer):
    def __init__(self, name, scope, pattern_template, transform_template=None):
        super().__init__(name, scope)

    def cache_file(self, element):
        return os.path.join(files.ndmake_dir(), "survey", self.name,
                            files.element_dirs(element), "matched_files")

    def read_mtimes(self, element):
        roster_old, roster_new = super().read_mtimes(element)
        if roster_old == 0 or roster_new == MAX_TIME:
            return 0, MAX_TIME

        # TODO Check files and get oldest and newest mtime.
        return min(roster_old, oldest_mtime), max(roster_new, newest_mtime)

    def read_result(self, element):
        file_list = super().read_result(element)
        files = file_list.splitlines()
        # TODO Do pattern matching and transformation on each filename and
        # return list of results.

    def survey(self, graph, survey, element):
        # TODO Do pattern matching in a smart way (dir-by-dir)
        # save file list
        # return list of transformed results.
        pass

    def dirname(self, element):
        return None # TODO Do something about this.


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
        assert False, "abstract method call"

    @property
    def is_surveyed(self):
        return isinstance(self.source, Survey)

    def is_demarcated(self, element):
        if not self.is_surveyed:
            return True
        survey = self.source
        return not not survey.is_result_available(element)

    def issubextent(self, other):
        if self is other:
            return True
        elif self.superextent is not None:
            if self.superextent.issubextent(other):
                return True
        return False


class SequenceExtentMixin:
    def __init__(self, *args, **kwargs): pass


class EnumeratedExtentMixin(SequenceExtentMixin):
    def raw_sequence(self, element):
        if isinstance(self.source, Survey):
            survey = self.source
            return survey.result(element)

        values_str = element.render_template(self.source)
        dprint_extent(self, "values string:", values_str)
        values = shlex.split(values_str)
        dprint_extent(self, "values:", ", ".join(values))
        return values


class ArithmeticExtentMixin(SequenceExtentMixin):
    def raw_sequence(self, element):
        if isinstance(self.source, Survey):
            survey = self.source
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

    def sequence(self, element):
        # raw_sequence() implemented by subclasses.
        return self.raw_sequence(element)

    def iterate(self, element):
        return iter(self.sequence(element))


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
    def sequence(self, element):
        return list(self.iterate(element))

    def iterate(self, element):
        super_values = frozenset(self.superextent.sequence(element))
        for value in self.raw_sequence(element):
            if value not in super_values:
                super_values = " ".join(self.superextent.sequence(element))
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
    def slice(self, element):
        if isinstance(self.source, Survey):
            survey = self.source
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

    def sequence(self, element):
        return self.superextent.sequence(element)[self.slice]

    def iterate(self, element):
        slice_ = self.slice(element)
        sliceargs = (slice_.start, slice_.stop, slice_.step)
        return iter(list(self.superextent.iterate(element))[slice(*sliceargs)])


#
# Caching mtimes
#

class ElementMTimeCache(dict):
    # A map from Element to (oldest_mtime, newest_mtime), with the ability to
    # aggregate mtimes from full elements to construct the mtimes for a partial
    # element (i.e. a subspace of the scope).
    def __init__(self, scope, mtimes_getter):
        self.scope = scope
        self.get_full_element_mtimes = mtimes_getter

    def __missing__(self, element):
        # Caller is responsible for ensuring that element is canonical.
        oldest, newest = MAX_TIME, 0
        for full_element, is_full in self.scope.iterate(element):
            if not is_full:
                return (..., ...) # No need to check further.
            if full_element in self:
                old, new = self[full_element]
            else:
                old, new = self.get_full_element_mtimes(full_element)
            if old == 0 or new == MAX_TIME:
                assert (old, new) == (0, MAX_TIME)
                return 0, MAX_TIME # No need to check further.
            oldest = min(oldest, old)
            newest = max(newest, new)
        return oldest, newest


#
# Templating support
#

class DatasetNameProxy:
    # An object to be bound to a dataset name when rendering a command
    # template.

    def __init__(self, dataset, default_element):
        self.__dataset = dataset
        default_element = dataset.scope.canonicalized_element(default_element)
        self.__default_element = default_element

    def __repr__(self):
        return "<DatasetNameProxy default={}>".format(repr(str(self)))

    def __filename_or_filenames(self, element):
        if self.__dataset.scope.is_full_element(element):
            return element.render_template(self.__dataset.filename_template)

        # We have a partial element; return a list.
        filenames = []
        for full_element, is_full in self.__dataset.scope.iterate(element):
            assert is_full
            filenames.append(full_element.
                             render_template(self.__dataset.filename_template))
        return " ".join(template.shellquote(name) for name in filenames)

    def __str__(self):
        return self.__filename_or_filenames(self.__default_element)

    def __call__(self, **kwargs):
        # Override and/or extend the default element with the kwargs.
        assigned_extents = []
        coords = {}
        for extent in self.__dataset.scope.extents:
            if extent.dimension in self.__default_element.space.dimensions:
                assigned_extents.append(extent)
                coords[extent.dimension] = \
                        self.__default_element[extent.dimension]
            if extent.dimension.name in kwargs:
                if extent not in assigned_extents:
                    assigned_extents.append(extent)
                coords[extent.dimension] = kwargs[extent.dimension.name]
        new_element = Element(Space(assigned_extents), coords)

        return self.__filename_or_filenames(new_element)
        
