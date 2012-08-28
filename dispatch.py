# Python 3

import collections
import functools
import heapq
import itertools
import queue
import warnings


__doc__ = """Cooperative multitasking in pure Python, using coroutines."""


class _EditablePriorityQueue():
    # See documentation for the heapq module.

    def __init__(self):
        self._heapqueue = []
        self._dic = {} # Keep track of queue entries.
        self._ordinal_generator = itertools.count(0)

    def __contains__(self, item):
        return item in self._dic

    def put(self, item, priority=0):
        ordinal = next(self._ordinal_generator)
        entry = [priority, ordinal, item]
        self._dic[item] = entry
        heapq.heappush(self._heapqueue, entry)

    def get(self):
        if not len(self._heapqueue):
            raise queue.Empty()

        while True:
            item = heapq.heappop(self._heapqueue)[-1]
            if item is not None:
                break
        del self._dic[item]
        return item

    def remove(self, item):
        if item in self._dic:
            self._dic[item][-1] = None
            del self._dic[item]


class _Tasklet():

    def __init__(self, tid, coroutine, priority=0):
        self.tid = tid
        self.coroutine = coroutine
        self.priority = priority

        self.retained_channels = {} # cid -> chan

        self.send_waiting_chan = None
        self.send_waiting_ref = None
        self.recv_waiting_chan = None
        self.select_waiting_descs = None # list of (io, chan)

    def run(self, sendval=None):
        return self.coroutine.send(sendval)

    def wait_on_send(self, chan, ref):
        self.send_waiting_chan = chan
        self.send_waiting_ref = ref

    def wait_on_recv(self, chan):
        self.recv_waiting_chan = chan
        chan.recv_queue[self.tid] = self

    def wait_on_select(self, kernel, descs):
        self.select_waiting_descs = []
        for io, cid in descs:
            chan = kernel.channels[cid]
            self.select_waiting_descs.append((io, chan))
            if io == Select.SEND:
                chan.select_send_queue[self.tid] = self
            else:
                chan.select_recv_queue[self.tid] = self
        kernel.select_queue[self.tid] = self

    def clear_waits(self, kernel):
        if self.tid in kernel.backlog_queue:
            kernel.backlog_queue.remove(self.tid)

        if self.send_waiting_chan is not None:
            self.send_waiting_chan = None
            self.send_waiting_ref = None
        elif self.recv_waiting_chan is not None:
            del self.recv_waiting_chan.recv_queue[self.tid]
            self.recv_waiting_chan = None
        elif self.select_waiting_descs is not None:
            for io, chan in self.select_waiting_descs:
                if io == Select.SEND:
                    del chan.select_send_queue[self.tid]
                else:
                    del chan.select_recv_queue[self.tid]
            del kernel.select_queue[self.tid]
            self.select_waiting_descs = None

    def release_channels(self, kernel):
        for chan in list(self.retained_channels.values()):
            chan.release(kernel, self)


class _Channel():

    def __init__(self, cid):
        self.cid = cid
        self.refcount = 0
        self.message_queue = collections.deque() # (sender_tid, message)

        self.message_ref_generator = itertools.count(1)

        # There is no send_queue because blocking senders' tids are saved in
        # message_queue.
        self.recv_queue = collections.OrderedDict() # tid -> tasklet
        self.select_send_queue = collections.OrderedDict() # tid -> tasklet
        self.select_recv_queue = collections.OrderedDict() # tid -> tasklet

    def retain(self, kernel, tasklet):
        if self.cid in tasklet.retained_channels:
            warnings.warn(("tasklet {tid} retaining already-retained " +
                           "channel {cid}").format(tid=tasklet.tid,
                                                   cid=self.cid))
        else:
            self.refcount += 1
            tasklet.retained_channels[self.cid] = self

    def release(self, kernel, tasklet):
        if self.cid in tasklet.retained_channels:
            self.refcount -= 1
            del tasklet.retained_channels[self.cid]
        else:
            warnings.warn(("tasklet {tid} releasing non-retained channel " +
                           "{cid}").format(tid=tasklet.tid, cid=self.cid))

        if self.refcount == 0:
            if len(self.message_queue):
                warnings.warn("deallocating non-empty channel {cid}".
                              format(cid=self.cid))
            del kernel.channels[self.cid]

    def enqueue(self, kernel, message, sender_tid=None):
        # We keep track of the sender's tid so that
        # 1) if the sender is blocking on the Send, it can be scheduled in
        #    the backlog once the message is received, and
        # 2) if the sender is waiting on the Send (while also being placed in
        #    the backlog queue), its waiting status can be cleared once the
        #    message is received.
        ref = next(self.message_ref_generator)
        self.message_queue.append((sender_tid, ref, message))
        kernel.nonempty_channels[self.cid] = self
        return ref

    def dequeue(self, kernel):
        try:
            sender_tid, ref, message = self.message_queue.popleft()
        except IndexError:
            raise queue.Empty
        else:
            if sender_tid is not None:
                if sender_tid in kernel.tasklets:
                    sender = kernel.tasklets[sender_tid]
                    # Check that the sender is still waiting on this send.
                    if (self.cid == sender.send_waiting_chan and
                        ref == sender.send_waiting_ref):
                        sender.clear_waits(kernel)
                    if sender_tid not in kernel.backlog_queue:
                        kernel.place_in_backlog(sender_tid)
            if self.is_empty():
                del kernel.nonempty_channels[self.cid]
            return message

    def is_empty(self):
        return not len(self.message_queue)


class _Kernel():
    def __init__(self):
        self.tid_generator = itertools.count(1)
        self.cid_generator = itertools.count(1)

        self.tasklets = {} # tid -> tasklet
        self.channels = {} # cid -> channel

        self.nonempty_channels = collections.OrderedDict() # cid -> channel
        self.select_queue = collections.OrderedDict() # tid -> tasklet
        self.backlog_queue = _EditablePriorityQueue() # tid
        self.async_queue = queue.Queue() # (cid, message)

    def start_with_tasklet(self, coroutine, priority=0):
        tid = self.new_tasklet(coroutine, priority)
        self.place_in_backlog(tid)
        self.runloop()

    def post_async_message(self, cid, message):
        self.async_queue.put((cid, message))

    def dispatch_async_messages(self):
        while True:
            try:
                cid, message = self.async_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.channels[cid].enqueue(kernel, message)

    def wait_for_async_message(self):
        cid, message = self.async_queue.get()
        self.channels[cid].enqueue(kernel, message)

    def new_tasklet(self, coroutine, priority):
        tasklet = _Tasklet(next(self.tid_generator), coroutine, priority)
        self.tasklets[tasklet.tid] = tasklet
        return tasklet.tid

    def terminate_all_tasklets(self):
        while self.tasklets:
            tid, tasklet = self.tasklets.popitem()
            tasklet.coroutine.close() # TODO Propagate exceptions, if any.
            tasklet.release_channels(self)

    def place_in_backlog(self, tid):
        priority = self.tasklets[tid].priority
        self.backlog_queue.put(tid, priority)

    def run_tasklet(self, tid, sendval=None):
        while True: # Keep running while message chain continues.
            tasklet = self.tasklets[tid]
            try:
                kcall = tasklet.run(sendval)
            except StopIteration: # Tasklet terminated successfully.
                tasklet.release_channels(self)
                del self.tasklets[tid]
                break
            except: # Tasklet terminated abnormally.
                tasklet.release_channels(self)
                del self.tasklets[tid]
                self.terminate_all_tasklets()
                raise

            if isinstance(kcall, _KernelCall):
                tid, sendval = kcall(self, tid)
                if tid is not None:
                    continue
                else:
                    break

            else: # Yielded a non-command. Yield to other tasklets.
                if kcall is not None:
                    # This is a programming error.
                    warnings.warn(("tasklet {tid} yielded unrecognized " +
                                   "value; discarding").format(tid=tid))
                self.place_in_backlog(tid)
                break

    def find_tasklet_to_run(self):
        # Return (tid, send_val).

        # 1) Unblock where chan has queued messages and a tasklet is blocking
        #    on Recv for the chan, or on Select for RECV on the chan.
        for i in range(len(self.nonempty_channels)):
            cid, chan = self.nonempty_channels.popitem(False)
            self.nonempty_channels[cid] = chan
            for tid, tasklet in chan.recv_queue.items():
                message = chan.dequeue(kernel)
                tasklet.clear_waits(self)
                return tid, message
            for tid, tasklet in chan.select_recv_queue.items():
                tasklet.clear_waits(self)
                return tid, (Select.RECV, cid)

        # 2) Unblock a tasklet blocking on Select for SEND on a chan where
        #    another tasklet is blocking on Recv for the chan, or on Select
        #    for RECV on the chan.
        for i in range(len(self.select_queue)):
            tid, tasklet = self.select_queue.popitem(False)
            self.select_queue[tid] = tasklet
            for cid in (cid for io, cid in tasklet.select_descs
                        if io == Select.SEND):
                chan = self.channels[cid]
                if len(chan.recv_queue) or len(chan.select_recv_queue):
                    tasklet.clear_waits(self)
                    return tid, (Select.SEND, cid)

        # 3) Resume a tasklet from the backlog, if any.
        try:
            tid = self.backlog_queue.get()
            self.tasklets[tid].clear_waits(self)
            return tid, None
        except queue.Empty:
            return None, None

    def runloop(self):
        while self.tasklets:
            self.dispatch_async_messages()
            tid, sendval = self.find_tasklet_to_run()
            if tid is not None:
                self.run_tasklet(tid, sendval)
            else:
                self.wait_for_async_message()


class _KernelCall():
    # Abstract base class.

    def __call__(self, kernel, tid):
        return tid, None # No-op.


class Spawn(_KernelCall):
    """Start a new tasklet.
    
    Usage: yield Spawn(tasklet[, priority]) # priority = 0 by default

    The new tasklet is scheduled to run immediately. The calling tasklet is
    placed in the backlog queue, and is resumed when there are no
    higher-priority tasklets to run.
    """

    def __init__(self, coroutine, priority=0):
        self.coroutine = coroutine
        self.priority = priority

    def __call__(self, kernel, tid):
        kernel.place_in_backlog(tid)
        new_tid = kernel.new_tasklet(self.coroutine, self.priority)
        return new_tid, None


class MakeChannel(_KernelCall):

    """Allocate a new channel.

    Usage: cid = yield MakeChannel()
    """

    def __call__(self, kernel, tid):
        chan = _Channel(next(kernel.cid_generator))
        chan.retain(kernel, kernel.tasklets[tid])
        kernel.channels[chan.cid] = chan
        return tid, chan.cid


class MakeChannels(_KernelCall):
    """Allocate multiple channels at once.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, kernel, tid):
        tasklet = kernel.tasklets[tid]
        ret = []
        for i in range(n):
            chan = _Channel(next(kernel.cid_generator))
            chan.retain(kernel, tasklet)
            kernel.channels[chan.cid] = chan
            ret.append(chan.cid)
        return tid, ret


class RetainChannel(_KernelCall):
    """Ensure that a channel is available for the lifetime of the tasklet.

    Usage: yield RetainChannel(cid)

    This should be called on channel ids passed from other tasklets, to prevent
    the channel from being deallocated.
    """

    def __init__(self, cid):
        self.cid = cid

    def __call__(self, kernel, tid):
        chan = kernel.channels[self.cid]
        tasklet = kernel.tasklets[tid]
        chan.retain(kernel, tasklet)
        return tid, None


class ReleaseChannel(_KernelCall):
    """Undo the effect of RetainChannel.

    Usage: yield ReleaseChannel(cid)

    This call should be used to release channels if many channels are created
    or used by a long-running tasklet. Normally, it is not necessary to
    explicitly release channels, as all channels retained by a tasklet are
    automatically released when the tasklet exits.
    """

    def __init__(self, cid):
        self.cid = cid

    def __call__(self, kernel, tid):
        chan = kernel.channels[self.cid]
        tasklet = kernel.tasklets[tid]
        chan.release(kernel, tasklet)
        return tid, None


class AsyncSender(_KernelCall):
    """Return a function to send asynchronous messages to a channel.

    Usage: sender = yield AsyncSender(cid)
           sender(message)

    The returned callable (sender) can be called from any thread.
    """

    def __init__(self, cid):
        self.cid = cid

    def __call__(self, kernel, tid):
        meth = functools.partial(kernel.post_async_message, self.cid)
        return tid, meth


class Select(_KernelCall):

    """Multiplex Send/Recv.

    Usage: io, cid = yield Select(descs[, block]) # block = True by default

    We shall call the pair (io, cid) a descriptor if io is either Select.SEND
    or Select.RECV and cid is a channel id.

    Select searches the given list of descriptors (desc) for one that is
    _ready_. If more than one descriptor is ready, the first one is returned
    in the order given in descs.

    SEND descriptors are ready if there is a tasklet blocked either on a Recv()
    or a Select(RECV) for the channel.

    RECV descriptors are considered ready if a message is pending on the
    channel (but not if a tasklet is blocked on a Select(SEND) on the channel).
    """

    SEND, RECV = "SEND", "RECV"

    def __init__(self, descs=[], block=True):
        self.descs = descs
        self.should_block = block

    def __call__(self, kernel, tid):
        tasklets_waiting_to_send = collections.OrderedDict()

        for io, cid in self.descs:
            chan = kernel.channels[cid]

            if io == self.SEND:
                # If a tasklet is blocking on Recv or Select(Recv) for the
                # channel, the current channel is ready to Send. 
                if len(chan.recv_queue) or len(chan.select_recv_queue):
                    return tid, (self.SEND, cid)

            else: # io == self.RECV:
                # If a message is available, the current channel is ready to
                # Recv.
                if not chan.is_empty():
                    return tid, (self.RECV, cid)

                # Otherwise, if there is a tasklet blocking on Select(Send),
                # we could switch to that tasklet if no other descriptor is
                # ready.
                for s_tid, s_tasklet in chan.select_send_queue:
                    tasklets_waiting_to_send[s_tid] = s_tasklet
                    break

        # None of the descriptors were ready.
        kernel.tasklets[tid].wait_on_select(kernel, self.descs)
        if not self.should_block:
            kernel.place_in_backlog(tid)

        # But we switch to a prospective sender if there were any.
        for s_tid, s_tasklet in tasklets_waiting_to_send:
            s_tasklet.clear_waits(kernel)
            return s_tid, (self.SEND, cid) # from blocked Select()

        # All concerned tasklets are suspended.
        return None, None


class Send(_KernelCall):

    """Send message through channel.
    
    Usage: yield Send(cid, message[, block]) # block = True by default
    """

    def __init__(self, cid, message, block=True):
        self.cid = cid
        self.message = message
        self.should_block = block

    def __call__(self, kernel, tid):
        chan = kernel.channels[self.cid]

        # If a tasklet is blocking on a Recv, unblock and switch to it, leaving
        # the current tasklet in the backlog.
        for r_tid, r_tasklet in chan.recv_queue.items():
            kernel.place_in_backlog(tid)
            r_tasklet.clear_waits(kernel)
            return r_tid, self.message # Return from Recv().

        # Otherwise, enqueue the message in the channel. The current tasklet
        # either blocks or is placed at the back of the backlog queue.
        ref = chan.enqueue(kernel, self.message, tid)
        kernel.tasklets[tid].wait_on_send(chan, ref)
        if not self.should_block:
            kernel.place_in_backlog(tid)

        # If there is a tasklet blocking on Select to Recv from the channel,
        # it is unblocked.
        for s_tid, s_tasklet in chan.select_recv_queue.items():
            s_tasklet.clear_waits(kernel)
            return s_tid, (Select.RECV, self.cid) # Return from Select().

        # Otherwise, all concerned tasklets have been suspended.
        return None, None


class Recv(_KernelCall):

    """Receive message from channel.
    
    Usage: message = yield Recv(cid[, block]) # block = True by default
    """

    def __init__(self, cid, block=True):
        self.cid = cid
        self.should_block = block

    def __call__(self, kernel, tid):
        chan = kernel.channels[self.cid]

        try:
            # If a message is available, return it.
            message = chan.dequeue(kernel)
            return tid, message # Return from Recv().

        except queue.Empty:
            # Otherwise, block or place in backlog.
            kernel.tasklets[tid].wait_on_recv(chan)
            if self.should_block:
                # If there is a tasklet blocking on Select to Send to the
                # channel, it is unblocked.
                for s_tid, s_tasklet in chan.select_send_queue.items():
                    s_tasklet.clear_waits(kernel)
                    return s_tid, (Select.SEND, self.cid) # from Select().
            else:
                kernel.place_in_backlog(tid)

            # Otherwise, all concerned tasklets have been suspended.
            return None, None


def start_with_tasklet(tasklet, priority=0):
    kern = _Kernel()
    kern.start_with_tasklet(tasklet, priority)
    # TODO We could conceivably have a mechanism to return a value ("Exit"
    # kernel call?). Note that the starting tasklet does not necessarily
    # survive till start_with_tasklet() returns.


def tasklet(coroutine):
    """A generator decorator indicating a tasklet.

    Actually returns the generator unmodified; use of this decorator is solely
    for the purpose of annotation.

    A tasklet is implemented as a coroutine. The following constructs can be
    used inside a coroutine:

        # Terminate successfully (raise an exception to terminate abnormally):
        return

        # Make a system call:
        result = yield NameOfKernelCall(*args, **kwargs)

        # Call a subroutine (which should also be a tasklet, but may return
        # arbitrary objects):
        retval = yield from Subroutine(*args, **kwargs)

    Tasklets should only allow exceptions to escape if the kernel, together
    with all other tasklets, is to be terminated.
    """
    return coroutine


def subtasklet(coroutine):
    """A generator decorator indicating a tasklet subroutine.

    Actually returns the generator unmodified; use of this decorator is solely
    for the purpose of annotation. 

    A subtasklet is similar to a tasklet, except that it is allowed to return
    a value (via a `return' statement), and can raise exceptions to be caught
    in the calling (sub)tasklet.
    """
    return coroutine

