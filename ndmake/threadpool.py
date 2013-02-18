import multiprocessing
import queue
import sys
import threading

from ndmake import dispatch


__doc__ = """Thread pool for use from tasklets."""


DEBUG = False
if DEBUG:
    def dprint(*args):
        print("threadpool:", *args)
else:
    def dprint(*args):
        pass


class _PoolThread(threading.Thread):

    def __init__(self, idle_notify_method):
        threading.Thread.__init__(self)
        self.daemon = True
        self.idle_notify_method = idle_notify_method
        self.task_queue = queue.Queue()

        self.start()

    def enqueue_task(self, func, context, reply_method, sequestered_threads):
        # To be called from the main thread.
        self.task_queue.put((func, context, reply_method, sequestered_threads))

    def finish(self):
        # To be called from the main thread.
        self.task_queue.put(...)

    def run(self):
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                dprint(self.name + ":", "idle")
                self.idle_notify_method(self)
                task = self.task_queue.get()  # Block.

            if task is ...:  # Sentinel.
                dprint(self.name + ":", "idle")
                self.idle_notify_method(self)
                break

            func, context, reply_method, sequestered_threads = task
            try:
                retval = func()
                exception = None
            except BaseException as e:
                retval = None
                exception = e
            result = context, retval, exception

            for sequestered_thread in sequestered_threads:
                dprint(sequestered_thread.name + ":", "idle (unsequestered)")
                self.idle_notify_method(sequestered_thread)

            reply_method(result)


@dispatch.subtasklet
def _wait_for_thread(all_threads, max_threads, idle_thread_chan):
    # Subroutine for threadpool().

    if len(all_threads) < max_threads:
        # Create a new thread.
        idle_notify_method = yield dispatch.AsyncSender(idle_thread_chan)
        thread = _PoolThread(idle_notify_method)
        all_threads.append(thread)
        dprint("started thread:", thread.name)

    # Wait for an idle thread.
    dprint("waiting for idle thread")
    thread = yield dispatch.Recv(idle_thread_chan)
    dprint("idle:", thread.name)

    return thread


@dispatch.tasklet
def threadpool(task_chan, max_threads=None):
    """Thread pool tasklet.

    Usage example (in a tasklet):

    # Create the thread pool:
    task_send_channel = yield dispatch.MakeChannel()
    yield dispatch.Spawn(threadpool(task_send_channel))

    # Enqueue a task:
    result_channel = yield dispatch.MakeChannel()
    yield dispatch.Send(task_send_channel, (func, context, result_channel,
                                            occupancy))
    # The Send will block until a thread is available (unless block=False is
    # given).

    # Retrieve the result:
    context, retval, exception = yield dispatch.Recv(result_channel)
    # The Recv will block until the result is available.
    # If func returned a value (retval), exception is None.
    # If func raised an exception, retval is None and exception is set.

    # Shut down the thread pool:
    finish_channel = yield dispatch.MakeChannel()
    yield dispatch.Send(task_send_channel, (..., None, finish_channel, None),
                        block=False)

    # Wait until thread pool has finished:
    context, retval, exception = yield dispatch.Recv(finish_channel)
    # In this example, context is None, retval is Ellipsis, exception is None.
    """

    if not max_threads:
        max_threads = multiprocessing.cpu_count()
    all_threads = []
    finished_threads = set()  # Threads that have finished their last task.
    idle_thread_chan = yield dispatch.MakeChannel()

    dprint("starting")

    try:  # On exception, we need to join all threads.

        while True:
            thread = yield from _wait_for_thread(all_threads, max_threads,
                                                 idle_thread_chan)

            func, context, reply_chan, occupancy = \
                yield dispatch.Recv(task_chan)

            if func is ...:  # Sentinel.
                finish_notify_chan = reply_chan
                finished_threads.add(thread)
                dprint("added to finished_threads:", thread.name)
                break

            # Start the task.
            # Ensure that we don't wait for more threads than will be
            # available.
            occupancy = max(1, min(max_threads, occupancy))
            extra_threads = []
            for i in range(occupancy - 1):
                extra_thread = yield from _wait_for_thread(all_threads,
                                                           max_threads,
                                                           idle_thread_chan)
                extra_threads.append(extra_thread)
            reply_method = yield dispatch.AsyncSender(reply_chan)
            dprint("enqueuing task on {} with {:d} extra threads:".
                   format(thread.name, len(extra_threads)),
                   ", ".join(sorted(t.name for t in extra_threads)))
            assert len(set(extra_threads)) == len(extra_threads)
            thread.enqueue_task(func, context, reply_method, extra_threads)

        # Shut down working threads.
        for thread in all_threads:
            thread.finish()

        # Wait for threads to finish.
        # Threads send themselves via the idle_thread_chan after each task is
        # completed, and once more after thread.finish() is called.
        # We place threads that have finished their final task in
        # finished_threads; threads that are already in finished_threads may
        # be joined as soon as they show up in idle_thread_chan again.
        if len(all_threads):
            dprint("waiting for threads to finish")
        while len(all_threads):
            thread = yield dispatch.Recv(idle_thread_chan)
            if thread in finished_threads:  # Thread terminated.
                thread.join()
                dprint("joined", thread.name)
                all_threads.remove(thread)
            else:  # Thread finished last task.
                finished_threads.add(thread)
                dprint("finished last task:", thread.name)

        # Signal that the thread pool has finished all tasks.
        dprint("exiting")
        yield dispatch.Send(finish_notify_chan, (context, ..., None))

    finally:
        # If an uncaught exception occurs, we make sure that the pool threads
        # are not leaked.
        if len(all_threads):
            dprint("terminating; waiting for threads to finish")
        for thread in all_threads:
            thread.finish()
        for thread in all_threads:
            thread.join()
            dprint("joined", thread.name)
