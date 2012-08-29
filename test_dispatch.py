# Python 3

import dispatch

@dispatch.tasklet
def a():
    recv_chans = []
    for i in range(10):
        send_chan = yield dispatch.MakeChannel()
        child_tasklet = yield dispatch.Spawn(b(send_chan))
        recv_chan = yield dispatch.MakeChannel()
        recv_chans.append(recv_chan)
        yield dispatch.Send(send_chan, i, block=False)
        yield dispatch.Send(send_chan, recv_chan, block=False)
    descs = [(dispatch.Select.RECV, c) for c in recv_chans]
    for i in range(10):
        _, chan = yield dispatch.Select(descs)
        m = yield dispatch.Recv(chan, block=False)
        assert m is not None
        print(m)

@dispatch.tasklet
def b(recv_chan):
    i = yield dispatch.Recv(recv_chan)
    send_chan = yield dispatch.Recv(recv_chan)
    yield dispatch.Send(send_chan, i, block=False)

dispatch.start_with_tasklet(a())

