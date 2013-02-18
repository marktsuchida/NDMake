from ndmake import debug
from ndmake import dispatch

dprint = debug.dprint_factory(__name__)


@dispatch.tasklet
def scatter(trigger_chan, notification_request_chan):
    # Blocks until a message is received on trigger_chan. Once received, sends
    # the message to all channels that have been received on
    # notification_request_chan up to then. Immediately sends the message to
    # any channels subsequently received from notification_request_chan.

    notification_chans = []

    while True:
        # Otherwise, we need to listen on two channels at the same time.
        dprint("tid {}".format((yield dispatch.GetTid())),
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

    dprint("tid {}".format((yield dispatch.GetTid())),
           "waiting for trigger")
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
def gather(trigger_chans, signal_combiner=None):
    trigger_chan_set = set(trigger_chans)
    if trigger_chan_set is trigger_chans:  # Just in case we were given a set.
        trigger_chan_set = trigger_chan_set.copy()

    yield dispatch.SetDaemon()

    messages = []
    while len(trigger_chan_set):
        dprint("tid {}".format((yield dispatch.GetTid())),
               "waiting on:", list(trigger_chan_set))
        descriptors = [(dispatch.Select.RECV, chan)
                       for chan in trigger_chan_set]
        io, trigger_chan = yield dispatch.Select(descriptors)
        dprint("tid {}".format((yield dispatch.GetTid())),
               "Select() returned:", trigger_chan)
        messages.append((yield dispatch.Recv(trigger_chan)))
        dprint("tid {}".format((yield dispatch.GetTid())),
               "message:", messages[-1])

        trigger_chan_set.remove(trigger_chan)

    dprint("tid {}".format((yield dispatch.GetTid())),
           "fired with messages:", messages)

    if signal_combiner is None:
        signal_combiner = lambda xs: xs[0] if xs else None
    return signal_combiner(messages)
