# Python 3

import collections
import functools
import heapq
import inspect
import itertools
import queue
import types
import warnings
import weakref

__doc__ = """Cooperative multitasking in pure Python, using coroutines."""


DEBUG = False
if DEBUG:
    def dprint(*args):
        print("dispatch: {}:".format(args[0]), *args[1:])
else:
    def dprint(*args): pass


# Coding conventions
# - Channels (_Channel instances) and channel handles (ChannelHandle instances)
#   are distinguished by naming variables chan and hchan.


class _OrderedSet(collections.deque):
    # Lazy implementation with only the used methods (add, remove, pop).
    # This deque-based implementation is much faster than an OrderedDict-based
    # implementation, at least for our purposes. Adding a set can speed up
    # add() but slows down pop() and others and does not contribute to overall
    # speedup.
    def add(self, m):
        if m not in self:
            self.append(m)

    def pop(self, last=True):
        m = super().pop() if last else self.popleft()
        return m


class _EditablePriorityQueue():
    # See documentation for the heapq module.

    def __init__(self):
        self._heapqueue = []
        self._dic = {} # Keep track of queue entries.
        self._ordinal_generator = itertools.count(0)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "_EditablePriorityQueue({})".format(repr(self._heapqueue))

    def __contains__(self, item):
        return item in self._dic

    def put(self, item, extra_info=None, priority=0):
        ordinal = next(self._ordinal_generator)
        entry = [priority, ordinal, item, extra_info]
        self._dic[item] = entry
        heapq.heappush(self._heapqueue, entry)

    def get(self):
        if not len(self._heapqueue):
            raise queue.Empty()

        while True:
            item, extra_info = heapq.heappop(self._heapqueue)[-2:]
            if item is not None:
                break
        del self._dic[item]
        return item, extra_info

    def remove(self, item):
        if item in self._dic:
            entry = self._dic[item]
            entry[-2] = entry[-1] = None
            del self._dic[item]


class _Tasklet():

    def __init__(self, kernel, tid, coroutine, priority=0, return_hchan=None):
        self.kernel = kernel
        self.tid = tid
        self.coroutine = coroutine
        self.priority = priority
        self.return_hchan = return_hchan
        self.is_daemon = False
        self.exited = False
        self.terminated = False

        self.send_waiting_hchan = None
        self.send_waiting_ref = None
        self.recv_waiting_hchan = None
        self.select_waiting_descs = None # list of (io, chan)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        status = ""
        if self.exited:
            status = " (exited)"
        elif self.terminated:
            status = " (terminated)"
        return "<Tasklet {} {}{}>".format(self.tid, self.coroutine.__name__,
                                          status)

    def run(self, sendval=None):
        return self.coroutine.send(sendval)

    def wait_on_send(self, hchan, ref):
        self.send_waiting_hchan = hchan
        self.send_waiting_ref = ref

    def wait_on_recv(self, hchan):
        self.recv_waiting_hchan = hchan
        hchan._chan.recv_queue.add(self)

    def wait_on_select(self, descs):
        self.select_waiting_descs = []
        for io, hchan in descs:
            self.select_waiting_descs.append((io, hchan))
            if io == Select.SEND:
                hchan._chan.select_send_queue.add(self)
            else:
                hchan._chan.select_recv_queue.add(self)
        self.kernel.select_queue.add(self)

    def clear_waits(self):
        if self in self.kernel.backlog_queue:
            self.kernel.backlog_queue.remove(self)

        if self.send_waiting_hchan is not None:
            self.send_waiting_hchan = None
            self.send_waiting_ref = None
        elif self.recv_waiting_hchan is not None:
            self.recv_waiting_hchan._chan.recv_queue.remove(self)
            self.recv_waiting_hchan = None
        elif self.select_waiting_descs is not None:
            for io, hchan in self.select_waiting_descs:
                if io == Select.SEND:
                    hchan._chan.select_send_queue.remove(self)
                else:
                    hchan._chan.select_recv_queue.remove(self)
            self.kernel.select_queue.remove(self)
            self.select_waiting_descs = None


class _Channel():
    # See also ChannelHandle. ChannelHandle is not a mere wrapper to keep the
    # _Channel implementation hidden from clients; it is a mechanism to tie
    # the lifetime of the channel to the existence of tasklets referencing it.
    # In general, holding a ChannelHandle is like holding a strong reference
    # to a channel, whereas holding a _Channel does not prevent the
    # corresponding ChannelHandle from being deallocated.

    def __init__(self, kernel, cid):
        self.kernel = kernel
        self.cid = cid
        self.message_queue = collections.deque() # (sender, ref, message)

        self.message_ref_generator = itertools.count(1)

        # There is no send_queue because blocking senders are saved in
        # message_queue.
        self.recv_queue = _OrderedSet() # tasklet
        self.select_send_queue = _OrderedSet() # tasklet
        self.select_recv_queue = _OrderedSet() # tasklet

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<Channel {}>".format(self.cid)

    def destroy(self, wref):
        assert wref is self.handle
        if len(self.message_queue):
            warnings.warn("deallocating non-empty channel {cid}".
                          format(cid=self.cid))
            self.kernel.channels.remove(self)
            # Note that channel may still remain in kernel.nonempty_channels.

    def enqueue(self, message, sender=None):
        # We keep track of the sender so that
        # 1) if the sender is blocking on the Send, it can be scheduled in
        #    the backlog once the message is received, and
        # 2) if the sender is waiting on the Send (while also being placed in
        #    the backlog queue), its waiting status can be cleared once the
        #    message is received.
        ref = next(self.message_ref_generator)
        self.message_queue.append((sender, ref, message))
        self.kernel.nonempty_channels.add(self)
        return ref

    def dequeue(self):
        try:
            sender, ref, message = self.message_queue.popleft()
        except IndexError:
            raise queue.Empty
        else:
            if sender is not None and sender in self.kernel.tasklets:
                # Check that the sender is still waiting on this send.
                if (sender.send_waiting_hchan is not None and
                    self is sender.send_waiting_hchan._chan and
                    ref == sender.send_waiting_ref):
                    sender.clear_waits()
                    if sender not in self.kernel.backlog_queue:
                        self.kernel.place_in_backlog(sender)
            if self.is_empty():
                self.kernel.nonempty_channels.remove(self)
            return message

    def is_empty(self):
        return not len(self.message_queue)


class _Kernel():
    def __init__(self):
        self.tid_generator = itertools.count(0)
        self.cid_generator = itertools.count(1)

        self.tasklets = set()
        self.channels = set()

        self.nonempty_channels = _OrderedSet() # channel
        self.select_queue = _OrderedSet() # tasklet
        self.backlog_queue = _EditablePriorityQueue() # tasklet
        self.async_queue = queue.Queue() # (hchan, message)

        self.tid0_exited = False

    def start_with_tasklet(self, coroutine, priority=0):
        tasklet = self.new_tasklet(coroutine, priority)
        self.place_in_backlog(tasklet)
        self.runloop()

    def post_async_message(self, hchan, message):
        self.async_queue.put((hchan, message))

    def dispatch_async_messages(self):
        while True:
            try:
                hchan, message = self.async_queue.get_nowait()
            except queue.Empty:
                break
            else:
                dprint(hchan._chan, "async send", message)
                hchan._chan.enqueue(message)

    def wait_for_async_message(self):
        hchan, message = self.async_queue.get()
        dprint(hchan._chan, "async send", message)
        hchan._chan.enqueue(message)

    def new_channel_handle(self):
        chan = _Channel(self, next(self.cid_generator))
        self.channels.add(chan)
        hchan = ChannelHandle(chan)
        return hchan

    def new_tasklet(self, coroutine, priority, return_hchan=None):
        # Catch a common error (forgetting to in clude a `yield' in the func).
        assert isinstance(coroutine, types.GeneratorType), \
                ("attempt to start tasklet with non-generator: {}".
                 format(coroutine))
        tasklet = _Tasklet(self, next(self.tid_generator), coroutine,
                           priority, return_hchan)
        self.tasklets.add(tasklet)
        return tasklet

    def terminate(self, tasklet):
        dprint(tasklet, "terminating")
        self.tasklets.remove(tasklet)
        tasklet.coroutine.close() # XXX Exceptions? Ignore for now.
        tasklet.terminated = True

    def terminate_all(self):
        for tasklet in list(self.tasklets):
            self.terminate(tasklet)

    def terminate_daemons(self):
        for tasklet in list(self.tasklets):
            if tasklet.is_daemon:
                self.terminate(tasklet)

    def place_in_backlog(self, tasklet, sendval=None):
        self.backlog_queue.put(tasklet, sendval, tasklet.priority)

    def run_tasklet(self, tasklet, sendval=None):
        while True: # Keep running while message chain continues.
            try:
                kcall = tasklet.run(sendval)

            except StopIteration as status: # Tasklet terminated successfully.
                dprint(tasklet, "exited")
                tasklet.exited = True
                self.tasklets.remove(tasklet)
                if tasklet.tid == 0: # Initial tasklet.
                    self.tid0_exited = True
                    self.terminate_daemons()

                return_value = status.value
                if tasklet.return_hchan is not None:
                    return_chan = tasklet.return_hchan._chan
                    ref = return_chan.enqueue(return_value, tasklet)
                    dprint(return_chan, tasklet, "returned", return_value)

                    # Are tasklets waiting for the return value?
                    keep_going = False
                    for s_tasklet in return_chan.select_recv_queue:
                        s_tasklet.clear_waits()
                        sendval = (Select.RECV, tasklet.return_hchan)
                        tasklet = s_tasklet
                        keep_going = True
                        break
                    if keep_going:
                        continue
                else:
                    dprint(tasklet, "return value discarded:", return_value)
                break

            except BaseException as e: # Tasklet terminated abnormally.
                dprint(tasklet, "raised", e.__class__.__name__)
                self.tasklets.remove(tasklet)
                self.terminate_all()
                raise

            if isinstance(kcall, _KernelCall):
                tasklet, sendval = kcall(self, tasklet)
                if tasklet is not None:
                    continue
                else:
                    break

            else: # Yielded a non-command. Yield to other tasklets.
                if kcall is not None:
                    # This is a programming error.
                    warnings.warn(("tasklet {tid} yielded unrecognized value; "
                                   "discarding").format(tid=tasklet.tid))
                self.place_in_backlog(tasklet)
                break

    def find_tasklet_to_run(self):
        # Return (tasklet, send_val).

        # 1) Unblock where chan has queued messages and a tasklet is blocking
        #    on Recv for the chan, or on Select for RECV on the chan.

        # Check each nonempty channel at most once, moving it to end of queue.
        for i in range(len(self.nonempty_channels)):
            chan = self.nonempty_channels.pop(last=False)
            if chan not in self.channels:
                continue
            self.nonempty_channels.add(chan)

            for tasklet in chan.recv_queue:
                message = chan.dequeue()
                dprint(chan, tasklet, "received", message)
                tasklet.clear_waits()
                return tasklet, message

            for tasklet in chan.select_recv_queue:
                tasklet.clear_waits()
                return tasklet, (Select.RECV, chan.handle)

        # 2) Unblock a tasklet blocking on Select for SEND on a chan where
        #    another tasklet is blocking on Recv for the chan, or on Select
        #    for RECV on the chan.

        # Check each tasklet at most once, moving it to end of queue.
        for i in range(len(self.select_queue)):
            tasklet = self.select_queue.pop(last=False)
            self.select_queue.add(tasklet)

            for hchan in (hchan for io, hchan in tasklet.select_waiting_descs
                          if io == Select.SEND):
                chan = hchan._chan
                if len(chan.recv_queue) or len(chan.select_recv_queue):
                    tasklet.clear_waits()
                    return tasklet, (Select.SEND, hchan)

        # 3) Resume a tasklet from the backlog, if any.
        try:
            tasklet, sendval = self.backlog_queue.get()
            tasklet.clear_waits()
            return tasklet, sendval
        except queue.Empty:
            return None, None

    def runloop(self):
        while len(self.tasklets):
            self.dispatch_async_messages()
            tasklet, sendval = self.find_tasklet_to_run()
            if tasklet is not None:
                self.run_tasklet(tasklet, sendval)
            else:
                self.wait_for_async_message()

        dprint("kernel", "exiting; {} tasklets, {} channels".
               format(next(self.tid_generator),
                      next(self.cid_generator) - 1))


class _KernelCall():
    # Abstract base class.

    def __call__(self, kernel, tasklet):
        return tasklet, None # No-op.


class Spawn(_KernelCall):
    """Start a new tasklet.

    Context (almost always) switches.
    
    Usage: yield Spawn(tasklet[, priority]) # priority = 0 by default

    The new tasklet is scheduled to run immediately. The calling tasklet is
    placed in the backlog queue, and is resumed when there are no
    higher-priority tasklets to run.
    """

    def __init__(self, coroutine, priority=0, return_chan=None):
        self.coroutine = coroutine
        self.priority = priority
        self.return_hchan = return_chan

    def __call__(self, kernel, tasklet):
        new_tasklet = kernel.new_tasklet(self.coroutine, self.priority,
                                         self.return_hchan)
        dprint(new_tasklet, "spawned by", tasklet)
        kernel.place_in_backlog(tasklet, TaskletHandle(new_tasklet))
        return new_tasklet, None


class MakeChannel(_KernelCall):

    """Allocate a new channel.

    Context does not switch.

    Usage: hchan = yield MakeChannel()
    """

    def __call__(self, kernel, tasklet):
        hchan = kernel.new_channel_handle()
        return tasklet, hchan


class MakeChannels(_KernelCall):
    """Allocate multiple channels at once.

    Context does not switch.

    Usage: hchans = yield MakeChannels(count)
    """

    def __init__(self, count):
        self.n = count

    def __call__(self, kernel, tasklet):
        hchans = [kernel.new_channel_handle() for i in range(n)]
        return tasklet, hchans


class AsyncSender(_KernelCall):
    """Return a function to send asynchronous messages to a channel.

    Context does not switch.

    Usage: sender = yield AsyncSender(hchan)
           sender(message)

    The returned callable (sender) can be called from any thread.
    """

    def __init__(self, hchan):
        self.hchan = hchan

    def __call__(self, kernel, tasklet):
        meth = functools.partial(kernel.post_async_message, self.hchan)
        return tasklet, meth


class Select(_KernelCall):

    """Multiplex Send/Recv.

    Context may switch.

    Usage: io, hchan = yield Select(descs[, block]) # block = True by default

    We shall call the pair (io, hchan) a descriptor if io is either
    Select.SEND or Select.RECV and hchan is a channel handle.

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

    def __call__(self, kernel, tasklet):
        tasklets_waiting_to_send = []

        for io, hchan in self.descs:
            chan = hchan._chan

            if io == self.SEND:
                # If a tasklet is blocking on Recv or Select(Recv) for the
                # channel, the current channel is ready to Send. 
                if len(chan.recv_queue) or len(chan.select_recv_queue):
                    return tasklet, (self.SEND, hchan)

            else: # io == self.RECV:
                # If a message is available, the current channel is ready to
                # Recv.
                if not chan.is_empty():
                    return tasklet, (self.RECV, hchan)

                # Otherwise, if there is a tasklet blocking on Select(Send),
                # we could switch to that tasklet if no other descriptor is
                # ready.
                for s_tasklet in chan.select_send_queue:
                    tasklets_waiting_to_send.append(s_tasklet)
                    break

        # None of the descriptors were ready.
        tasklet.wait_on_select(self.descs)
        if not self.should_block:
            kernel.place_in_backlog(tasklet)

        # But we switch to a prospective sender if there were any.
        for s_tasklet in tasklets_waiting_to_send:
            s_tasklet.clear_waits()
            # from blocked Select()
            return s_tasklet, (self.SEND, s_tasklet.send_waiting_hchan)

        # All concerned tasklets are suspended.
        return None, None


class Send(_KernelCall):

    """Send message through channel.

    Context may switch.
    
    Usage: yield Send(hchan, message[, block]) # block = True by default
    """

    def __init__(self, hchan, message, block=True):
        self.hchan = hchan
        self.message = message
        self.should_block = block

    def __call__(self, kernel, tasklet):
        chan = self.hchan._chan

        # If a tasklet is blocking on a Recv, unblock and switch to it, leaving
        # the current tasklet in the backlog.
        for r_tasklet in chan.recv_queue:
            kernel.place_in_backlog(tasklet)
            r_tasklet.clear_waits()
            return r_tasklet, self.message # Return from Recv().

        # Otherwise, enqueue the message in the channel. The current tasklet
        # either blocks or is placed at the back of the backlog queue.
        ref = chan.enqueue(self.message, tasklet)
        dprint(chan, tasklet, "sent", self.message)
        tasklet.wait_on_send(self.hchan, ref)
        if not self.should_block:
            kernel.place_in_backlog(tasklet)

        # If there is a tasklet blocking on Select to Recv from the channel,
        # it is unblocked.
        for s_tasklet in chan.select_recv_queue:
            s_tasklet.clear_waits()
            # Return from Select():
            return s_tasklet, (Select.RECV, self.hchan)

        # Otherwise, all concerned tasklets have been suspended.
        return None, None


class Recv(_KernelCall):

    """Receive message from channel.

    Context may switch.
    
    Usage: message = yield Recv(hchan[, block]) # block = True by default
    """

    def __init__(self, hchan, block=True):
        self.hchan = hchan
        self.should_block = block

    def __call__(self, kernel, tasklet):
        try:
            # If a message is available, return it.
            message = self.hchan._chan.dequeue()
            dprint(self.hchan._chan, tasklet, "received", message)
            return tasklet, message # Return from Recv().

        except queue.Empty:
            # Otherwise, block or place in backlog.
            tasklet.wait_on_recv(self.hchan)
            if self.should_block:
                # If there is a tasklet blocking on Select to Send to the
                # channel, it is unblocked.
                for s_tasklet in self.hchan._chan.select_send_queue:
                    s_tasklet.clear_waits()
                    # from Select():
                    return s_tasklet, (Select.SEND, self.hchan)
            else:
                kernel.place_in_backlog(tasklet)

            # Otherwise, all concerned tasklets have been suspended.
            return None, None


class GetTid(_KernelCall):

    """Get the current tasklet id.

    Context does not switch.

    Usage: tid = yield GetTid()
    """

    def __init__(self): pass

    def __call__(self, kernel, tasklet):
        tid = tasklet.tid
        return tasklet, tid


class SetDaemon(_KernelCall):

    """Set the daemon flag for the current tasklet.

    Tasklets whose daemon flag is set will be killed when the main (initial)
    tasklet exits. If SetDaemon(True) is called when the main tasklet has
    already exited, the calling tasklet is immediately terminated.

    Context does not switch.

    Usage: yield SetDaemon(flag) # or
           yield SetDaemon() # Same as flag = True.
    """

    def __init__(self, make_daemon=True):
        self.flag = make_daemon

    def __call__(self, kernel, tasklet):
        tasklet.is_daemon = self.flag
        if kernel.tid0_exited:
            kernel.terminate(tasklet)
            return None, None
        return tasklet, None


class TaskletHandle():
    """A handle for the opaque tasklet object.
    
    TaskletHandle objects may be passed between tasklets, via Send() or
    Spawn().
    """

    def __init__(self, _tasklet):
        self._tasklet = _tasklet

    @property
    def tid(self):
        """The tasklet's unique id."""
        return self._tasklet.tid


class ChannelHandle():
    """A handle for the opaque channel object.
    
    ChannelHandle objects may be passed between tasklets, via Send() or
    Spawn().
    """

    def __init__(self, _chan):
        self._chan = _chan
        _chan.handle = weakref.ref(self, _chan.destroy)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "<ChannelHandle {}>".format(self._chan.cid)

    @property
    def cid(self):
        """The channel's unique id."""
        return self._chan.cid


def start_with_tasklet(tasklet, priority=0):
    kern = _Kernel()
    kern.start_with_tasklet(tasklet, priority)
    # TODO We could conceivably have a mechanism to return a value ("Exit"
    # kernel call?). Note that the starting tasklet does not necessarily
    # survive till start_with_tasklet() returns.


def tasklet(coroutine_func):
    """A generator function decorator indicating a tasklet.

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
    assert inspect.isgeneratorfunction(coroutine_func), \
            "@tasklet applied to non-generator function"
    return coroutine_func


def subtasklet(coroutine_func):
    """A generator function decorator indicating a tasklet subroutine.

    Actually returns the generator unmodified; use of this decorator is solely
    for the purpose of annotation. 

    A subtasklet is similar to a tasklet, except that it is allowed to return
    a value (via a `return' statement), and can raise exceptions to be caught
    in the calling (sub)tasklet.
    """
    assert inspect.isgeneratorfunction(coroutine_func), \
            "@subtasklet applied to non-generator function"
    return coroutine_func

