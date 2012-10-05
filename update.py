from collections import OrderedDict
import depgraph
import dispatch
import os.path
import shlex
import time
import debug

dprint = debug.dprint_factory(__name__, True)

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

    message = yield dispatch.Recv(trigger_chan)
    dprint("tid {}".format((yield dispatch.GetTid())),
           "fired with message:", message)

    for notification_chan in notification_chans:
        yield dispatch.Send(notification_chan, message, block=False)

    del notification_chans
    yield dispatch.SetDaemon()

    while True:
        notification_chan = yield dispatch.Recv(notification_request_chan)
        dprint("tid {}".format((yield dispatch.GetTid())),
               "post-hoc message:", message)
        yield dispatch.Send(notification_chan, message, block=False)


@dispatch.tasklet
def demultiplex(trigger_chans, signal_combiner=max):
    trigger_chan_set = set(trigger_chans)
    if trigger_chan_set is trigger_chans: # Just in case we were given a set.
        trigger_chan_set = trigger_chan_set.copy()

    yield dispatch.SetDaemon()

    messages = []
    while len(trigger_chan_set):
        descriptors = [(dispatch.Select.RECV, chan)
                       for chan in trigger_chan_set]
        io, trigger_chan = yield dispatch.Select(descriptors)
        messages.append((yield dispatch.Recv(trigger_chan)))

        trigger_chan_set.remove(trigger_chan)

    dprint("tid {}".format((yield dispatch.GetTid())),
           "fired with messages:", messages)
    return signal_combiner(messages)


class Update:
    def __init__(self, graph):
        decorator_map = OrderedDict()
        decorator_map[depgraph.Dataset] = DatasetUpdateRuntime
        decorator_map[depgraph.Computation] = ComputationUpdateRuntime
        decorator_map[depgraph.ValuesSurvey] = ValuesSurveyUpdateRuntime
        decorator_map[depgraph.RangeSurvey] = RangeSurveyUpdateRuntime
        decorator_map[depgraph.SliceSurvey] = SliceSurveyUpdateRuntime
        decorator_map[depgraph.Vertex] = VertexUpdateRuntime
        decorator_map[...] = depgraph.RuntimeDecorator

        self.graph = depgraph.DynamicGraph(graph, decorator_map)

    @dispatch.subtasklet
    def _get_notification_request_chan(self, vertex):
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
    def update_vertex(self, vertex):
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
        downstream_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(multiplex(downstream_chan, request_chan))
        yield dispatch.Spawn(self._update_vertex(vertex),
                             return_chan=downstream_chan)

    @dispatch.tasklet
    def _update_vertex(self, vertex):
        # Update prerequisites.
        yield from self.update_vertices(self.graph.parents_of(vertex))

        # Perform the update action.
        dprint("tid {}".format((yield dispatch.GetTid())),
               "visiting", vertex)
        action_notification_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(do_update(vertex),
                             return_chan=action_notification_chan)
        action_result = yield dispatch.Recv(action_notification_chan)

        return action_result

    @dispatch.tasklet
    def update_vertices(self, vertices):
        vertices = list(self.graph.runtime(v) for v in vertices)

        for vertex in vertices:
            yield dispatch.Spawn(self.update_vertex(vertex))

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
            yield dispatch.Recv(completion_chan) # Wait for completion.


class NotUpToDateException(Exception):
    # Exception raised when a not-up-to-date Computation is encountered.
    pass


@dispatch.tasklet
def do_update(vertex, **options):
    if False: yield None # Make it a generator.
    # Vertex is runtime vertex.
    return (yield from vertex.update_all_elements(**options))


class VertexUpdateRuntime(depgraph.RuntimeDecorator):
    def __init__(self, graph, vertex):
        super().__init__(graph, vertex)

        self.update_started = False # Prevent duplicate update.
        self.notification_request_chan = None # Request chan for mux.

    @dispatch.tasklet
    def update_all_elements(self, **options):
        assert False, "abstract method call"


class DatasetUpdateRuntime(VertexUpdateRuntime):
    def __init__(self, graph, dataset):
        super().__init__(graph, dataset)
        self.file_stats_cache = {} # element -> (oldest_mtime, newest_mtime)

    @dispatch.tasklet
    def update_all_elements(self, **options):
        if False: yield None # Make it a generator.
        # This is sufficient to check that all files are present in the case
        # of a dataset that is a source vertex. For non-source vertices,
        # this simply returns a cached value.
        newest_mtime = self.newest_mtime()
        if newest_mtime == MAX_TIME: # XXX For now.
            raise NotUpToDateException(self)
        return newest_mtime

    def check_file_stats(self, element=depgraph.Element()):
        element = self.scope.canonical_element(element)
        if element not in self.file_stats_cache:
            oldest, newest = MAX_TIME, 0
            scope = self.graph.runtime(self.scope)
            for full_element in scope.iterate(element):
                if full_element in self.file_stats_cache:
                    old, new = self.file_stats_cache[full_element]
                    if (old, new) == (0, MAX_TIME):
                        oldest, newest = 0, MAX_TIME
                        break
                    mtime = old # == new
                else:
                    full_element = self.graph.runtime(full_element)
                    filename = (full_element.
                                render_template(self.filename_template))
                    if not os.path.exists(filename):
                        oldest, newest = 0, MAX_TIME
                        break
                    mtime = os.path.getmtime(filename)

                if mtime < oldest:
                    oldest = mtime
                if mtime > newest:
                    newest = mtime

            self.file_stats_cache[element] = (oldest, newest)

    def oldest_mtime(self, element=depgraph.Element()):
        # The element must be in a valid subspace of the dataset's scope.
        # If files are missing, 0 is returned.
        self.check_file_stats(element)
        return self.file_stats_cache[element][0]

    def newest_mtime(self, element=depgraph.Element()):
        # The element must be in a valid subspace of the dataset's scope.
        # If files are missing, a large number is returned.
        self.check_file_stats(element)
        return self.file_stats_cache[element][1]


class ComputationUpdateRuntime(VertexUpdateRuntime):
    def __init__(self, graph, computation):
        super().__init__(graph, computation)

    @dispatch.tasklet
    def update_all_elements(self, **options):
        for element in self.scope.iterate(self.graph):
            if self.is_element_locally_up_to_date(element):
                continue
            raise NotUpToDateException(self) # XXX For now.

    def is_element_locally_up_to_date(self, element):
        # The element must belong to our scope.
        assert element.scope.is_compatible_space(self.scope)
        inputs = self.graph.parents_of(self)
        outputs = self.graph.children_of(self)
        newest_input_mtime = (max(input.newest_mtime(element)
                                  for input in inputs)
                              if len(inputs) else 0)
        oldest_output_mtime = (min(output.oldest_mtime(element)
                                   for output in outputs)
                               if len(outputs) else 1)
        return newest_input_mtime < oldest_output_mtime


class ValuesSurveyUpdateRuntime(VertexUpdateRuntime):
    pass

class RangeSurveyUpdateRuntime(VertexUpdateRuntime):
    pass

class SliceSurveyUpdateRuntime(VertexUpdateRuntime):
    pass

