import dispatch
import depgraph
from collections import OrderedDict


DEBUG = True
if DEBUG:
    def dprint(*args):
        print("update: {}:".format(args[0]), *args[1:])
else:
    def dprint(*args): pass


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

    for notification_chan in notification_chans:
        yield dispatch.Send(notification_chan, message, block=False)

    del notification_chans
    yield dispatch.SetDaemon()

    while True:
        notification_chan = yield dispatch.Recv(notification_request_chan)
        yield dispatch.Send(notification_chan, message, block=False)


@dispatch.tasklet
def demultiplex(trigger_chans, notification_chan, signal=Ellipsis):
    trigger_chan_set = set(trigger_chans)
    if trigger_chan_set is trigger_chans: # Just in case we were given a set.
        trigger_chan_set = trigger_chan_set.copy()

    while len(trigger_chan_set):
        descriptors = [(dispatch.Select.RECV, chan)
                       for chan in trigger_chan_set]
        io, trigger_chan = yield dispatch.Select(descriptors)
        yield dispatch.Recv(trigger_chan) # Ignore the message

        trigger_chan_set.remove(trigger_chan)

    yield dispatch.Send(notification_chan, signal)



class VertexUpdateRuntime(depgraph.VertexStateRuntime):
    def __init__(self, graph, vertex):
        super().__init__(graph, vertex)

        self.update_started = False
        self.notification_request_chan = None


class Update:
    def __init__(self, graph, action):
        proxy_map = OrderedDict()
        proxy_map[depgraph.Vertex] = VertexUpdateRuntime
        proxy_map[...] = depgraph.RuntimeProxy
        self.graph = depgraph.DynamicGraph(graph, proxy_map)
        self.action = action

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

        # Start updating prerequisites.
        prereqs = self.graph.parents_of(vertex)
        for prereq_vertex in prereqs:
            yield dispatch.Spawn(self.update_vertex(prereq_vertex))

        # Set up to receive notification as prerequisites are completed.
        # And implicitly trigger recursive update of prerequisites.
        prereq_notification_chans = [] # Can't use generator expression here.
        for prereq_vertex in prereqs:
            notification_chan = (yield from
                                 self._get_notification_chan(prereq_vertex))
            prereq_notification_chans.append(notification_chan)

        if len(prereq_notification_chans):
            upstream_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(demultiplex(prereq_notification_chans,
                                             upstream_chan))
            yield dispatch.Recv(upstream_chan) # Wait for completion signal.

        # Perform the update action.
        dprint("tid {}".format((yield dispatch.GetTid())),
               "visiting", vertex)
        self.action(vertex)

        # Notify our completion.
        yield dispatch.Send(downstream_chan, Ellipsis, block=False)

    @dispatch.tasklet
    def update_sinks(self):
        sinks = self.graph.sinks()
        for sink_vertex in sinks:
            yield dispatch.Spawn(self.update_vertex(sink_vertex))

        sink_notification_chans = [] # Can't use generator expression here.
        for sink_vertex in sinks:
            notification_chan = (yield from
                                 self._get_notification_chan(sink_vertex))
            sink_notification_chans.append(notification_chan)
        if len(sink_notification_chans):
            completion_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(demultiplex(sink_notification_chans,
                                             completion_chan))
            yield dispatch.Recv(completion_chan) # Wait for completion.

