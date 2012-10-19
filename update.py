from collections import OrderedDict
import os.path
import shlex
import subprocess
import sys
import time

import debug
import depgraph
import dispatch
import files
import template
import threadpool

dprint_mux = debug.dprint_factory(__name__, False)
dprint_traverse = debug.dprint_factory(__name__, False)
dprint_mtime = debug.dprint_factory(__name__, False)
dprint = debug.dprint_factory(__name__, False)
def strfmtime(mtime):
    return (time.strftime("%Y%m%dT%H%M%S", time.localtime(mtime)) +
            ".{:04d}".format(round(mtime % 1 * 10000)))

# Traverse a depgraph to bring a vertex (or set of vertices) up to date.
#
# Traversal is at whole-vertex granularity; subspace updates are not supported
# by this module.
#
# Starting from a set of vertices (to be brought up to date), the depgraph is
# traversed in the descendant-to-ancestor direction to set up a graph of
# tasklets that update (or otherwise act on) the vertices. Once the source
# vertices are reached, forward (ancestor-to-descendant) traversal commences,
# during which it is ensured that vertex action tasklets are called only after
# all parent vertex action tasklets have successfully completed.
#
# The second, forward phase can be modified by the keep_going option (named
# after `make --keep-going'). If keep_going is False, the first exception
# raised in an action tasklet causes all tasklets to be terminated. If
# keep_going is set to True, update continues for vertices that are not
# descendants of the vertex whose tasklet raised an exception.
# XXX Or just propagate this info through mux/demux and do nothing in
# descendant action tasklet?
#
# Other variations of traversal are implemented as different action tasklets
# (or their options).
#
# The standard update action does the following.
# - If the vertex is a Computation, each element is checked for up-to-dateness:
#   an element is up to date iff the newest input file is older than the oldest
#   output file. If the element is not up to date, the command for that element
#   is run. After the command is run, outputs are checked for consistency. 
#   (See XXX for some more details.)
# - If the vertex is a Dataset with no parent (a source vertex), each element
#   is checked to ensure all files are present. If the vertex is a Dataset with
#   a producer, nothing is done.
# - If the vertex is a Survey, action depends on the specific type of the
#   survey. For command surveys, the command is run and the result is cached
#   (XXX and written to a file?). For filename surveys, filename pattern
#   matching is performed (with consideration of the timestamp of the parent
#   Computation, if any) and the result is cached.
#
# The following modifiers can be applied to the standard update action.
# - ignore_errors: If True, non-zero command exit statuses are ignored. Checks
#   for output consistency are performed, however.
# - query_only: If True, commands are never run (except for command surveyer
#   commands). Upon the first encountered Computation element that is not up to
#   date, a NotUpToDateException is raised and the traversal is terminated.
#
# The dry-run update action works similarly to the standard update action,
# except that it never runs commands, and that it can continue the forward
# traversal even if vertex spaces are not determined due to surveyer whose
# input is not available. The action tasklets thus run in one of two modes
# (definite-space and indefinite-space). In the indefinite-space mode,
# iteration over vertex elements is not performed and instead a single action
# (usually printing a status line) is performed.
#
# In the implementation, the standard and dry-run action tasklets are combined
# as they share a considerable portion of code.


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


@dispatch.tasklet
def multiplex(trigger_chan, notification_request_chan):
    # Blocks until a message is received on trigger_chan. Once received, sends
    # the message to all channels that have been received on
    # notification_request_chan up to then. Immediately sends the message to
    # any channels subsequently received from notification_request_chan.

    notification_chans = []

    while True:
        # Otherwise, we need to listen on two channels at the same time.
        dprint_mux("tid {}".format((yield dispatch.GetTid())),
                   "waiting for trigger on {} or request on {}".
                   format(trigger_chan, notification_request_chan))
        io, chan = yield dispatch.Select([(dispatch.Select.RECV,
                                           trigger_chan),
                                          (dispatch.Select.RECV,
                                           notification_request_chan)])

        if chan is trigger_chan:
            # Proceed to receive message.
            break

        if chan is notification_request_chan:
            notification_chan = yield dispatch.Recv(notification_request_chan)
            notification_chans.append(notification_chan)

    dprint_mux("tid {}".format((yield dispatch.GetTid())),
               "waiting for trigger")
    message = yield dispatch.Recv(trigger_chan)
    dprint_mux("tid {}".format((yield dispatch.GetTid())),
               "fired with message:", message)

    for notification_chan in notification_chans:
        yield dispatch.Send(notification_chan, message, block=False)

    del notification_chans
    yield dispatch.SetDaemon()

    while True:
        notification_chan = yield dispatch.Recv(notification_request_chan)
        dprint_mux("tid {}".format((yield dispatch.GetTid())),
                   "post-hoc message:", message)
        yield dispatch.Send(notification_chan, message, block=False)


@dispatch.tasklet
def demultiplex(trigger_chans, signal_combiner=None):
    trigger_chan_set = set(trigger_chans)
    if trigger_chan_set is trigger_chans: # Just in case we were given a set.
        trigger_chan_set = trigger_chan_set.copy()

    yield dispatch.SetDaemon()

    messages = []
    while len(trigger_chan_set):
        dprint_mux("tid {}".format((yield dispatch.GetTid())),
                   "waiting on:", list(trigger_chan_set))
        descriptors = [(dispatch.Select.RECV, chan)
                       for chan in trigger_chan_set]
        io, trigger_chan = yield dispatch.Select(descriptors)
        dprint_mux("tid {}".format((yield dispatch.GetTid())),
                   "Select() returned:", trigger_chan)
        messages.append((yield dispatch.Recv(trigger_chan)))
        dprint_mux("tid {}".format((yield dispatch.GetTid())),
                   "message:", messages[-1])

        trigger_chan_set.remove(trigger_chan)

    dprint_mux("tid {}".format((yield dispatch.GetTid())),
               "fired with messages:", messages)

    if signal_combiner is None:
        signal_combiner = lambda xs: xs[0] if xs else None
    return signal_combiner(messages)


class MissingFileException(Exception): pass
class CalledProcessError(Exception): pass


class Update:
    def __init__(self, graph):
        decorator_map = OrderedDict()
        decorator_map[depgraph.Dataset] = DatasetUpdateRuntime
        decorator_map[depgraph.Computation] = ComputationUpdateRuntime
        decorator_map[depgraph.Survey] = SurveyUpdateRuntime

        decorator_map[depgraph.IntegerTripletCommandSurveyer] = \
                IntegerTripletCommandSurveyerRuntime
        decorator_map[depgraph.ValuesCommandSurveyer] = \
                ValuesCommandSurveyerRuntime
        decorator_map[depgraph.FilenameSurveyer] = FilenameSurveyerRuntime

        decorator_map[...] = depgraph.RuntimeDecorator

        self.graph = depgraph.DynamicGraph(graph, decorator_map)

    @dispatch.subtasklet
    def _get_notification_request_chan(self, vertex):
        # Return the unique channel attached to each vertex to which
        # notification requests can be sent. Notification is requested by
        # sending the handle to the notification channel to the channel
        # returned by this subtasklet. All notification channels receive a
        # signal exactly once, after the vertex has been updated or as soon as
        # the request is made, whichever comes later.
        request_chan = vertex.notification_request_chan
        if request_chan is None:
            request_chan = yield dispatch.MakeChannel()
            vertex.notification_request_chan = request_chan
        return request_chan

    @dispatch.subtasklet
    def _get_notification_chan(self, vertex):
        # Return a new channel that will receive notification of completion
        # of update for the given vertex. The channel will recieve a signal
        # even if update has already been completed by the time this subtasklet
        # is called.
        request_chan = (yield from
                        self._get_notification_request_chan(vertex))

        # Create and register a new notification channel.
        notification_chan = yield dispatch.MakeChannel()
        yield dispatch.Send(request_chan, notification_chan, False)
        return notification_chan

    @dispatch.tasklet
    def update_vertex(self, vertex, **options):
        vertex = self.graph.runtime(vertex)

        # Prevent duplicate execution.
        if vertex.update_started:
            return
        vertex.update_started = True

        # Set up a channel by which completion notification channels can be
        # registered.
        request_chan = (yield from
                        self._get_notification_request_chan(vertex))

        # Set up notification for our completion.
        completion_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(multiplex(completion_chan, request_chan))
        yield dispatch.Spawn(self._update_vertex(vertex, **options),
                             return_chan=completion_chan)

    @dispatch.tasklet
    def _update_vertex(self, vertex, **options):
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing upward:", vertex)

        # Update prerequisites.
        parents = self.graph.parents_of(vertex)
        yield from self._update_vertices(parents, **options)

        # Perform the update action.
        dprint_traverse("tid {}".format((yield dispatch.GetTid())),
                        "traversing downward:", vertex)
        completion_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(vertex.update_all_elements(**options),
                             return_chan=completion_chan)
        yield dispatch.Recv(completion_chan)

    @dispatch.tasklet
    def _update_vertices(self, vertices, **options):
        vertices = list(self.graph.runtime(v) for v in vertices)

        for vertex in vertices:
            yield dispatch.Spawn(self.update_vertex(vertex, **options))

        # Set up to receive notification as vertices are completed.
        # And implicitly trigger recursive update of prerequisites.
        notification_chans = [] # Can't use generator expression here.
        for vertex in vertices:
            notification_chan = (yield from
                                 self._get_notification_chan(vertex))
            notification_chans.append(notification_chan)
        if len(notification_chans):
            completion_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(demultiplex(notification_chans),
                                 return_chan=completion_chan)
            yield dispatch.Recv(completion_chan)

    @dispatch.tasklet
    def update_vertices(self, vertices, **options):
        new_options = options.copy()
        if options.get("parallel", False):
            task_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(threadpool.threadpool(task_chan))
            new_options["threadpool"] = task_chan

        yield from self._update_vertices(vertices, **new_options)

        if "threadpool" in new_options:
            finish_chan = yield dispatch.MakeChannel()
            yield dispatch.Send(task_chan, (..., None, finish_chan, None),
                                block=False)
            yield dispatch.Recv(finish_chan)


class NotUpToDateException(Exception):
    # Exception raised when a not-up-to-date Computation is encountered.
    pass


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


class VertexUpdateRuntime(depgraph.RuntimeDecorator):
    def __init__(self, graph, vertex):
        super().__init__(graph, vertex)

        self.update_started = False # Prevent duplicate update.
        self.notification_request_chan = None # Request chan for mux.

        self.mtimes = ElementMTimeCache(self.graph.runtime(self.scope),
                                        self.read_mtimes)

    def read_mtimes(self):
        assert False, "abstract method call"

    def newest_mtime(self, element=depgraph.Element()):
        element = self.scope.canonicalized_element(element)
        _, newest = self.mtimes[element]
        return newest

    def oldest_mtime(self, element=depgraph.Element()):
        element = self.scope.canonicalized_element(element)
        oldest, _ = self.mtimes[element]
        return oldest

    @dispatch.tasklet
    def update_element(self, element, is_full, **options):
        if 0: yield
        assert False, "abstract method call"

    @dispatch.tasklet
    def update_all_elements(self, **options):
        dprint(self, "updating all elements")
        completion_chans = []
        scope = self.graph.runtime(self.scope)
        for element, is_full in scope.iterate():
            completion_chan = yield dispatch.MakeChannel()
            completion_chans.append(completion_chan)
            yield dispatch.Spawn(self.update_element(element, is_full,
                                                     **options),
                                 return_chan=completion_chan)
        all_complete_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(demultiplex(completion_chans),
                             return_chan=all_complete_chan)
        yield dispatch.Recv(all_complete_chan)


class DatasetUpdateRuntime(VertexUpdateRuntime):
    def read_mtimes(self, element):
        filename = element.render_template(self.filename_template)
        try:
            mtime = os.path.getmtime(filename)
            dprint_mtime("mtime", strfmtime(mtime), filename)
        except FileNotFoundError:
            dprint_mtime("mtime", "missing", filename)
            return 0, MAX_TIME
        return mtime, mtime # oldest, newest

    @dispatch.tasklet
    def update_element(self, element, is_full, **options):
        if 0: yield
        dprint(self,
               "updating {} element:".format("full" if is_full else "partial"), 
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
            parents = self.graph.parents_of(self)
            for parent in parents:
                if isinstance(parent.static_object, depgraph.Computation):
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
            full_element = depgraph.Element(self.scope, coords)
            filename = full_element.render_template(self.filename_template)
            dirname = os.path.dirname(filename)
            while unassigned_marker in dirname:
                dirname = os.path.dirname(dirname)
                if not dirname:
                    return None
            return dirname

    def name_proxy(self, element):
        return DatasetNameProxy(self, element)


class ComputationUpdateRuntime(VertexUpdateRuntime):
    @dispatch.tasklet
    def update_element(self, element, is_full, **options):
        if 0: yield
        dprint(self,
               "updating {} element:".format("full" if is_full else "partial"), 
               element)

        if not is_full:
            # XXX Print if requested.
            self.mtimes[element] = (..., ...)
            return

        parents = self.graph.parents_of(self)
        newest_input_mtime = 0
        for parent in parents:
            assert isinstance(parent, DatasetUpdateRuntime)
            parent_newest_mtime = parent.newest_mtime(element)
            newest_input_mtime = max(newest_input_mtime, parent_newest_mtime)

        previous_starttime, previous_finishtime = self.read_mtimes(element)

        if previous_finishtime < newest_input_mtime:
            # In this case we know we are out of date and need not check the
            # dataset files.
            pass
        else:
            children = self.graph.children_of(self)
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
        self.create_dirs_for_output(element)
        self.touch_starttime_stamp(element)
        yield from self.run_command(element, **options)
        self.touch_finishtime_stamp(element)

    def read_mtimes(self, element):
        startstamp = os.path.join(self.cache_dir(element), "start")
        try:
            start = os.path.getmtime(startstamp)
            dprint_mtime("mtime", strfmtime(start), startstamp)
        except FileNotFoundError:
            dprint_mtime("mtime", "missing", startstamp)
            start = 0

        finishstamp = os.path.join(self.cache_dir(element), "finish")
        try:
            finish = os.path.getmtime(finishstamp)
            dprint_mtime("mtime", strfmtime(finish), finishstamp)
        except FileNotFoundError:
            dprint_mtime("mtime", "missing", finishstamp)
            finish = MAX_TIME

        return start, finish

    def create_dirs_for_output(self, element, **options):
        for child in self.graph.children_of(self):
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
    def run_command(self, element, **options):
        if 0: yield

        # Bind input and output dataset names.
        io_vertices = (self.graph.parents_of(self) +
                       self.graph.children_of(self))
        dataset_name_proxies = dict((v.name, v.name_proxy(element))
                                    for v in io_vertices
                                    if hasattr(v, "name_proxy"))
        # Note: filename-surveyed output datasets' names are not available in
        # the command template.

        command = element.render_template(self.command_template,
                                          extra_names=dataset_name_proxies)

        def task_func():
            print(command)
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


class SurveyUpdateRuntime(VertexUpdateRuntime):
    def __init__(self, graph, survey):
        super().__init__(graph, survey)
        self.results = {}

    def is_result_available(self, element):
        # Iff we've been updated, the results are stored in self.results.
        element = self.scope.canonicalized_element(element)
        return element in self.results

    def result(self, element):
        element = self.scope.canonicalized_element(element)
        return self.results[element]

    def read_mtimes(self, element):
        surveyer = self.graph.runtime(self.surveyer)
        return surveyer.read_mtimes(element)

    @dispatch.tasklet
    def update_element(self, element, is_full, **options):
        if 0: yield
        dprint(self,
               "updating {} element:".format("full" if is_full else "partial"), 
               element)

        if not is_full:
            self.results[element] = ... # Not really necessary.
            return

        parents = self.graph.parents_of(self)
        newest_input_mtime = 0
        for parent in parents:
            assert isinstance(parent, DatasetUpdateRuntime)
            newest_input_mtime = max(newest_input_mtime,
                                     parent.newest_mtime(element))

        surveyer = self.graph.runtime(self.surveyer)

        previous_oldest, previous_newest = surveyer.read_mtimes(element)
        if newest_input_mtime <= previous_oldest:
            # We are up to date; load previous results.
            self.results[element] = surveyer.read_result(element)
            self.mtimes[element] = (previous_oldest, previous_newest)
            return

        # We are not up to date, so do the survey.
        self.results[element] = surveyer.survey(self, element)
        self.mtimes[element] = surveyer.read_mtimes(element)


class SurveyerRuntime(depgraph.RuntimeDecorator):
    def read_mtimes(self, element):
        filename = self.cache_file(element)
        try:
            mtime = os.path.getmtime(filename)
            dprint_mtime("mtime", strfmtime(mtime), filename)
        except FileNotFoundError:
            dprint_mtime("mtime", "missing", filename)
            return 0, MAX_TIME
        return mtime, mtime # oldest, newest

    def read_result(self, element):
        with open(self.cache_file(element)) as file:
            result_text = file.read()
        return result_text


class CommandSurveyerRuntime(SurveyerRuntime):
    def cache_file(self, element):
        return os.path.join(files.ndmake_dir(), "survey", self.name,
                            files.element_dirs(element), "command_output")

    def read_result(self, element):
        result_text = super().read_result(element)
        return self.convert_result(result_text)

    def survey(self, survey, element):
        # Bind input dataset names.
        parents = self.graph.parents_of(survey)
        dataset_name_proxies = dict((parent.name, parent.name_proxy(element))
                                    for parent in parents
                                    if isinstance(parent,
                                                  DatasetUpdateRuntime))

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
        

class IntegerTripletCommandSurveyerRuntime(CommandSurveyerRuntime):
    def convert_result(self, result_text):
        try:
            result = tuple(int(v) for v in result_text.split())
            assert len(result) in range(1, 4)
        except:
            raise ValueError("command output does not conform to range or "
                             "slice argument format (1-3 integers required)")
        return result


class ValuesCommandSurveyerRuntime(CommandSurveyerRuntime):
    def convert_result(self, result_text):
        return result_text.splitlines()


class FilenameSurveyerRuntime(SurveyerRuntime):
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

    def survey(self, survey, element):
        # TODO Do pattern matching in a smart way (dir-by-dir)
        # save file list
        # return list of transformed results.
        pass

    def dirname(self, element):
        return None # TODO Do something about this.


class DatasetNameProxy:
    # An object to be bound to a dataset name when rendering a command
    # template.

    def __init__(self, dataset_runtime, default_element):
        self.__dataset = dataset_runtime
        default_element = dataset_runtime.scope. \
                canonicalized_element(default_element)
        self.__default_element = default_element

    def __repr__(self):
        return "<DatasetNameProxy default={}>".format(repr(str(self)))

    def __filename_or_filenames(self, element):
        dataset = self.__dataset

        if dataset.scope.is_full_element(element):
            return element.render_template(dataset.filename_template)

        # We have a partial element; return a list.
        filenames = []
        scope = dataset.graph.runtime(dataset.scope)
        for full_element, is_full in scope.iterate(element):
            assert is_full
            filenames.append(full_element.
                             render_template(dataset.filename_template))
        return " ".join(template.shellquote(name) for name in filenames)

    def __str__(self):
        return self.__filename_or_filenames(self.__default_element)

    def __call__(self, **kwargs):
        dataset, element = self.__dataset, self.__default_element

        # Override and/or extend the default element with the kwargs.
        assigned_extents = []
        coords = {}
        for extent in dataset.scope.extents:
            if extent.dimension in element.space.dimensions:
                assigned_extents.append(extent)
                coords[extent.dimension] = element[extent.dimension]
            if extent.dimension.name in kwargs:
                if extent not in assigned_extents:
                    assigned_extents.append(extent)
                coords[extent.dimension] = kwargs[extent.dimension.name]
        new_element = depgraph.Element(depgraph.Space(assigned_extents),
                                       coords)

        return self.__filename_or_filenames(new_element)
        
