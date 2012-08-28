# Python 3

import dispatch

@dispatch.tasklet
def a():
    recv_cids = []
    for i in range(10):
        send_cid = yield dispatch.MakeChannel()
        yield dispatch.Spawn(b(send_cid))
        recv_cid = yield dispatch.MakeChannel()
        recv_cids.append(recv_cid)
        yield dispatch.Send(send_cid, i, block=False)
        yield dispatch.Send(send_cid, recv_cid, block=False)
        yield dispatch.ReleaseChannel(send_cid)
    descs = [(dispatch.Select.RECV, c) for c in recv_cids]
    for i in range(10):
        _, cid = yield dispatch.Select(descs)
        m = yield dispatch.Recv(cid, block=False)
        assert m is not None
        print(m)

@dispatch.tasklet
def b(recv_cid):
    i = yield dispatch.Recv(recv_cid)
    send_cid = yield dispatch.Recv(recv_cid)
    yield dispatch.Send(send_cid, i, block=False)

dispatch.start_with_tasklet(a())

