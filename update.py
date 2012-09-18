import dispatch

# A start of an ndmake implementation using dispatch.py.


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

        if io == dispatch.Select.EOF:
            if chan is trigger_chan:
                # This is a programming error.
                raise RuntimeError("trigger channel closed before message "
                                   "received")
            if chan is notification_request_chan:
                # Stop listening on the closed notification_request_chan, and
                # proceed to block until message received on trigger_chan.
                break

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

    while True:
        try:
            notification_chan = yield dispatch.Recv(notification_request_chan)
        except EOFError:
            return
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
        if io == dispatch.Select.EOF:
            # This is a programming error.
            raise RuntimeError("trigger channel closed before message "
                               "received")
        yield dispatch.Recv(trigger_chan) # Ignore the message

    yield dispatch.Send(notification_chan, signal)



class VertexRuntime(depgraph.RuntimeProxy):
    def __init__(self, vertex):
        super().__init__(vertex)

        self.update_started = False
        self.notification_request_chan = None

class Update():
    def __init__(self, graph, action):
        self.graph = graph
        self.action = action

    @dispatch.subtasklet
    def get_notification_request_chan(self, vertex):
        request_chan = vertex.notification_request_chan
        if request_chan is None:
            request_chan = yield dispatch.MakeChannel()
            vertex.notification_request_chan = request_chan
        return request_chan

    @dispatch.subtasklet
    def get_notification_chan(self, vertex):
        # Return a new channel that will receive notification of completion
        # of update for the given vertex. The channel will recieve a signal
        # even if update has already been completed by the time this subtasklet
        # is called.

        request_chan = (yield from
                        self.get_notification_request_chan(vertex))

        # Create and register a new notification channel.
        notification_chan = yield dispatch.MakeChannel()
        yield dispatch.Send(request_chan, notification_chan, False)
        return notification_chan

    @dispatch.tasklet
    def update_vertex(self, vertex):
        # Prevent duplicate execution.
        if vertex.update_started:
            return
        vertex.update_started = True

        # Set up a channel by which completion notification channels can be
        # registered.
        request_chan = (yield from
                        self.get_notification_request_chan(vertex))

        # Set up notification for our completion.
        downstream_chan = yield dispatch.MakeChannel()
        yield dispatch.Spawn(multiplex(downstream_chan, request_chan))

        # Start updating prerequisites.
        for prereq_vertex in self.graph.parents_of(vertex):
            yield dispatch.Spawn(self.update_vertex(prereq_vertex))

        # Set up to receive notification as prerequisites are completed.
        # And implicitly trigger recursive update of prerequisites.
        prereq_notification_chans = \
                [(yield from self.get_notification_chan(prereq_vertex))
                 for prereq_vertex in self.graph.parents_of(vertex)]
        if len(prereq_notification_chans):
            upstream_chan = yield dispatch.MakeChannel()
            yield dispatch.Spawn(demultiplex(prereq_notification_chans,
                                             upstream_chan))
            yield dispatch.Recv(upstream_chan) # Wait for completion signal.

        # Perform the update action.
        self.action(vertex)

        # Notify our completion.
        yield dispatch.Send(downstream_chan, Ellipsis, block=False)

