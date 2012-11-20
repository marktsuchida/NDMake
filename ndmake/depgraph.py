import errno
import functools
import itertools
import multiprocessing
import os, os.path
import shlex
import subprocess
import sys

from ndmake import debug
from ndmake import dispatch
from ndmake import files
from ndmake import mtime
from ndmake import mux
from ndmake import space
from ndmake import template
from ndmake import threadpool


dprint = debug.dprint_factory(__name__)
dprint_traverse = debug.dprint_factory(__name__, "traverse")
dprint_update = debug.dprint_factory(__name__, "update")
dprint_undemarcated = debug.dprint_factory(__name__, "undemarcated")
dprint_unlink = debug.dprint_factory(__name__, "unlink")


#
# Exceptions
#

class UpdateException(Exception): pass
class MissingFileException(UpdateException): pass
class CalledProcessError(UpdateException): pass


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
        self.template_environment = template.Environment()

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

    def vertex_by_name(self, name, type_):
        """Return a vertex with the given name and type."""
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
            if vertex is existing_vertex:
                return existing_id
            raise KeyError("a {} vertex named {} already exists".
                           format(vertex.namespace_type.__name__,
                                  vertex.name))

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

    def simplify_by_transitive_reduction(self):
        visited_ids = set()

        def visit(vertex_id, previous_vertex_id):
            visited_ids.add(vertex_id)

            # Remove shortcut edges from visited ancestors.
            for parent_id in list(self._parents.get(vertex_id, [])):
                if parent_id is previous_vertex_id:
                    continue

                if parent_id in visited_ids:
                    # Remove this edge if its tail is a Survey.
                    if isinstance(self._id_vertex_map[parent_id], Survey):
                        self._parents[vertex_id].remove(parent_id)
                        self._children[parent_id].remove(vertex_id)

            # Proceed to children.
            for child_id in list(self._children.get(vertex_id, [])):
                # Children may be removed during iteration.
                if child_id in self._children.get(vertex_id, []):
                    visit(child_id, vertex_id)

            visited_ids.remove(vertex_id)

        for vertex_id in self._id_vertex_map:
            visit(vertex_id, None)

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
    def update_vertices(self, vertices, options):
        for vertex in vertices:
            yield dispatch.Spawn(vertex.update(self, options))

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
    def update_vertices_with_threadpool(self, vertices, options):
        if options.get("parallel", False):
            jobs = options.get("jobs")
            if not jobs or jobs < 1:
                jobs = multiprocessing.cpu_count()
                options["jobs"] = jobs
            task_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(threadpool.threadpool(task_chan, jobs))
            options["threadpool"] = task_chan

        yield from self.update_vertices(vertices, options)

        if "threadpool" in options:
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

    def __repr__(self):
        return "<{} \"{}\">".format(type(self).__name__, self.name)

    def __str__(self):
        return "{} {}".format(type(self).__name__.lower(), self.name)

    @property
    def namespace_type(self):
        if isinstance(self, Dataset):
            return Dataset
        if isinstance(self, Computation):
            return Computation
        if isinstance(self, Survey):
            return Survey

    @dispatch.tasklet
    def update_all_elements(self, graph, options):
        # This method implements element-by-element update. Subclasses can
        # override this method to do a full-scope check before calling super().
        dprint_update(self, "updating all elements")
        completion_chans = []
        for element, is_full in self.scope.iterate():
            completion_chan = yield dispatch.MakeChannel()
            completion_chans.append(completion_chan)
            yield dispatch.Spawn(self.update_element(graph, element, is_full,
                                                     options),
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
    def update(self, graph, options):
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
        yield dispatch.Spawn(self._update(graph, options),
                             return_chan=completion_chan)

    @dispatch.tasklet
    def _update(self, graph, options):
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing upward:", self)

        # Update prerequisites.
        parents = graph.parents_of(self)
        yield from graph.update_vertices(parents, options)

        # Perform the update action.
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing downward:", self)
        if options.get("print_traversed_vertices", True):
            print("starting check/update of {}".format(self))
        completion_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(self.update_all_elements(graph, options),
                             return_chan=completion_chan)
        yield dispatch.Recv(completion_chan)
        if options.get("print_traversed_vertices", True):
            print("finished check/update of {}".format(self))

    def invalidate_up_to_date_cache(self, graph, element):
        # Propagate computation status invalidation to descendants.
        # This generic implementation just propagates the call; Computation
        # overrides this to implement the actual invalidation.
        for child in graph.children_of(self):
            child.invalidate_up_to_date_cache(graph, element)


#
# Concrete Vertices
#

class Dataset(Vertex):
    def __init__(self, graph, name, scope, filename_template):
        super().__init__(graph, name, scope)
        self.filename_template = filename_template

        self.mtimes = space.Cache(self.scope, self.read_mtimes, mtime.extrema)
        persistence_path = os.path.join(files.ndmake_dir(), "data", self.name)
        self.mtimes.set_persistence(mtime.reader, mtime.writer,
                                    path=persistence_path, filename="mtimes",
                                    level=2)

    def __str__(self):
        return "data {}".format(self.name)

    def render_filename(self, element):
        dict_ = {"__name__": self.name}
        return element.render_template(self.filename_template,
                                       extra_names=dict_)

    def read_mtimes(self, element):
        return mtime.get(self.render_filename(element))

    @dispatch.tasklet
    def update_all_elements(self, graph, options):
        if options.get("survey_only", False):
            return

        oldest_mtime, newest_mtime = self.mtimes[space.Element()]
        if oldest_mtime > 0:
            dprint_update(self, "all elements up to date")
            if options.get("cache", False):
                self.mtimes.save_to_file()
            return

        yield from super().update_all_elements(graph, options)
        if options.get("cache", False):
            self.mtimes.save_to_file()

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, options):
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        if not is_full:
            dprint_undemarcated("undemarcated", self, element)
            return

        if options.get("survey_only", False):
            return

        oldest_mtime, newest_mtime = self.mtimes[element]

        if oldest_mtime == 0 or newest_mtime == mtime.MAX_TIME:
            # There are missing files.
            # Unless this is a dry run or a keep-going run, we raise an error.
            # XXX For now, we raise an error unconditionally.
            filename = self.render_filename(element)
            parents = graph.parents_of(self)
            for parent in parents:
                if isinstance(parent, Computation):
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
        return
        yield

    def clean(self, graph, element, cache_only=False):
        if cache_only:
            del self.mtimes[element]
        else:
            self.delete_files(element)

        for parent in graph.parents_of(self):
            if isinstance(parent, Computation):
                parent.invalidate_up_to_date_cache(graph, element)

    def delete_files(self, element):
        for full_element, is_full in self.scope.iterate(element):
            assert is_full
            filename = self.render_filename(full_element)
            dprint_unlink("deleting", filename)
            if os.path.exists(filename):
                os.unlink(filename)
        del self.mtimes[element]

    def dirname(self, element):
        element = self.scope.canonicalized_element(element)
        if self.scope.is_full_element(element):
            return os.path.dirname(self.render_filename(element))
        else:
            # Given a partial element, we still want to be able to create
            # directories whose names are fixed. There might be a better way
            # to do this, but for now, we empirically find the common prefix
            # path by setting unassigned dimensions to different values.
            coords1, coords2 = {}, {}
            for extent in self.scope.extents:
                if extent.dimension in element.space.dimensions:
                    value = element[extent.dimension]
                    coords1[extent.dimension] = value
                    coords2[extent.dimension] = value
                else:
                    coords1[extent.dimension] = extent.value_type(123456)
                    coords2[extent.dimension] = extent.value_type(654321)
            full_element1 = space.Element(self.scope, coords1)
            full_element2 = space.Element(self.scope, coords2)
            dirname1 = os.path.dirname(self.render_filename(full_element1))
            dirname2 = os.path.dirname(self.render_filename(full_element2))
            while dirname1 != dirname2:
                dirname1 = os.path.dirname(dirname1)
                dirname2 = os.path.dirname(dirname2)
                if not dirname1 or not dirname2:
                    return None
            return dirname1

    def create_dirs(self, element):
        dirname = self.dirname(element)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def name_proxy(self, element):
        return DatasetNameProxy(self, element)


class Computation(Vertex):
    def __init__(self, graph, name, scope, command_template, occupancy=1):
        super().__init__(graph, name, scope)
        self.command_template = command_template
        self.occupancy = occupancy

        is_up_to_date = functools.partial(self.is_up_to_date, graph)
        self.statuses = space.Cache(self.scope, is_up_to_date, all)
        persistence_path = os.path.join(files.ndmake_dir(), "compute",
                                        self.name)
        self.statuses.set_persistence(lambda s: bool(int(s)),
                                      lambda b: "1" if b else "0",
                                      path=persistence_path, filename="status",
                                      level=2)

    def __str__(self):
        return "compute {}".format(self.name)

    def render_command(self, graph, element):
        # Bind input and output dataset names.
        io_vertices = (graph.parents_of(self) + graph.children_of(self))
        dataset_name_proxies = dict((v.name, v.name_proxy(element))
                                    for v in io_vertices
                                    if hasattr(v, "name_proxy"))
        # Note: filename-surveyed output datasets' names are not available in
        # the command template.
        dict_ = dataset_name_proxies
        dict_["__name__"] = self.name
        return element.render_template(self.command_template,
                                       extra_names=dict_)

    def is_up_to_date(self, graph, element):
        oldest_child_mtime, _ = mtime.extrema(child.mtimes[element]
                                              for child
                                              in graph.children_of(self))
        if oldest_child_mtime > 0:
            _, newest_input_mtime = mtime.extrema(parent.mtimes[element]
                                                  for parent
                                                  in graph.parents_of(self)
                                                  if isinstance(parent,
                                                                Dataset))
            if newest_input_mtime <= oldest_child_mtime:
                return True
        return False

    @dispatch.tasklet
    def update_all_elements(self, graph, options):
        if options.get("survey_only", False):
            return

        status = self.statuses[space.Element()]
        if status:
            dprint_update(self, "all elements up to date")
            if options.get("cache", False):
                self.statuses.save_to_file()
            return

        yield from super().update_all_elements(graph, options)
        if options.get("cache", False):
            self.statuses.save_to_file()

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, options):
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        if not is_full:
            dprint_undemarcated("undemarcated", self, element)
            return

        if options.get("survey_only", False):
            return

        if self.statuses[element]:
            return

        yield from self.execute(graph, element, options)

    def clean(self, graph, element, cache_only=False):
        self.invalidate_up_to_date_cache(graph, element)
        if not cache_only:
            for child in graph.children_of(self):
                child.delete_files(element)

    @dispatch.subtasklet
    def execute(self, graph, element, options):
        del self.statuses[element]
        self.invalidate_up_to_date_cache(graph, element)
        for child in graph.children_of(self):
            child.delete_files(element)
            child.create_dirs(element)
        yield from self.run_command(graph, element, options)

    @dispatch.subtasklet
    def run_command(self, graph, element, options):
        command = self.render_command(graph, element)
        print_command = options.get("print_executed_commands", False)

        def task_func():
            # Avoid print(), which apparently flushes the output between the
            # string and the newline.
            if print_command:
                sys.stdout.write(command + "\n")

            try:
                with subprocess.Popen(command, shell=True) as proc:
                    proc.wait()
                    return proc.returncode
            except OSError as e:
                if e.errno == errno.E2BIG: # Argument list too long.
                    # Fall back to piping to shell, which will help if the long
                    # command is e.g. a here document.
                    with subprocess.Popen("/bin/sh",
                                          stdin=subprocess.PIPE) as shellproc:
                        shellproc.communicate(command.encode())
                        return shellproc.returncode
                raise

        outputs_are_valid = False
        try:
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
                # For now, we raise unconditionally. XXX If requested, outputs
                # should be considered valid even if retval != 0.
                raise CalledProcessError("command returned exit status of "
                                         "{:d}: {}".format(retval, command))
            else:
                outputs_are_valid = True

        finally:
            # If the command failed, we need to delete any output files, which
            # may be corrupt.
            if not outputs_are_valid:
                for child in graph.children_of(self):
                    child.delete_files(element)

    def invalidate_up_to_date_cache(self, graph, element):
        del self.statuses[element]
        super().invalidate_up_to_date_cache(graph, element)


class Survey(Vertex):
    def __init__(self, graph, name, scope, surveyer):
        super().__init__(graph, name, scope)
        self.surveyer = surveyer

        self.results = {}

        self.mtimes = space.Cache(self.scope, self.surveyer.read_mtimes,
                                  mtime.extrema)
        persistence_path = os.path.join(files.ndmake_dir(), "survey",
                                        self.name)
        self.mtimes.set_persistence(mtime.reader, mtime.writer,
                                    path=persistence_path, filename="mtimes",
                                    level=2)

    def __str__(self):
        return "survey for {}".format(self.name)

    def is_result_available(self, element):
        # Iff we've been updated, the results are stored in self.results.
        element = self.scope.canonicalized_element(element)
        return element in self.results

    def result(self, element):
        element = self.scope.canonicalized_element(element)
        return self.results[element]

    @dispatch.tasklet
    def update_all_elements(self, graph, options):
        yield from super().update_all_elements(graph, options)
        if options.get("cache", False):
            self.mtimes.save_to_file()

    @dispatch.tasklet
    def update_element(self, graph, element, is_full, options):
        dprint_update(self,
                      "updating {} element:".
                      format("full" if is_full else "partial"),
                      element)

        if not is_full:
            dprint_undemarcated("undemarcated", self, element)
            return

        for parent in graph.parents_of(self):
            if isinstance(parent, Dataset):
                # Command survey with input(s).
                break
            if isinstance(parent, Computation):
                # Filename survey with producer.
                break
        else:
            # We have a command survey with no inputs or a filename survey on a
            # non-computed dataset: always run survey.
            yield from self.execute(graph, element, options)
            return

        our_mtime, _ = self.mtimes[element]
        if our_mtime > 0:
            if self.surveyer.mtimes_include_files:
                self.load_result(element)
                return

            _, newest_input_mtime = mtime.extrema(parent.mtimes[element]
                                                  for parent
                                                  in graph.parents_of(self)
                                                  if isinstance(parent,
                                                                Dataset))
            if newest_input_mtime <= our_mtime:
                self.load_result(element)
                return

        yield from self.execute(graph, element, options)

    @dispatch.subtasklet
    def execute(self, graph, element, options):
        self.surveyer.delete_files(element, delete_surveyed_files=False)
        del self.mtimes[element]
        self.invalidate_up_to_date_cache(graph, element)

        # Bind input dataset names.
        dataset_name_proxies = dict((parent.name, parent.name_proxy(element))
                                    for parent in graph.parents_of(self)
                                    if isinstance(parent, Dataset))
        dict_ = dataset_name_proxies
        dict_["__name__"] = self.name

        self.results[element] = self.surveyer.run_survey(self, element,
                                                         dict_, options)
        return
        yield

    def load_result(self, element):
        self.results[element] = self.surveyer.load_result(element)

    def delete_files(self, graph, element):
        self.surveyer.delete_files(element, delete_surveyed_files=True)
        del self.mtimes[element]

    def create_dirs(self, element):
        self.surveyer.create_dirs(element)


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
        # Use a summarized version of __quoted_filenames().
        filenames = self.__filename_list(self.__default_element)
        if len(filenames) > 3:
            summary = " ".join((shlex.quote(filenames[0]),
                                "...",
                                shlex.quote(filenames[-1])))
        else:
            summary = " ".join(shlex.quote(name) for name in filenames)

        return "<DatasetNameProxy default={}>".format(repr(summary))

    def __filename_list(self, element):
        if self.__dataset.scope.is_full_element(element):
            return [self.__dataset.render_filename(element)]

        # We have a partial element.
        return list(self.__dataset.render_filename(full_element)
                    for full_element, is_full
                    in self.__dataset.scope.iterate(element)
                    if is_full)

    def __quoted_filenames(self, element):
        return " ".join(shlex.quote(name)
                        for name in self.__filename_list(element))

    def __str__(self):
        return self.__quoted_filenames(self.__default_element)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == other

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
        new_element = space.Element(space.Space(assigned_extents), coords)

        return self.__quoted_filenames(new_element)
        
