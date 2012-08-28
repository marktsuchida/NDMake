# Python 3

import dispatch
import multiprocessing
import queue
import sys
import threading
import warnings


__doc__ = """Thread pool for use from tasklets."""


class _PoolThread(threading.Thread):

    def __init__(self, idle_notify_method):
        threading.Thread.__init__(self)
        self.daemon = True
        self.idle_notify_method = idle_notify_method
        self.task_queue = queue.Queue()

        self.start()

    def enqueue_task(self, func, context, reply_method):
        # To be called from the main thread.
        self.task_queue.put((func, context, reply_method))

    def finish(self):
        # To be called from the main thread.
        self.task_queue.put(Ellipsis)

    def run(self):
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                self.idle_notify_method(self)
                task = self.task_queue.get() # Block.

            if task is Ellipsis: # Sentinel.
                self.idle_notify_method(self)
                break

            func, context, reply_method = task
            try:
                retval = func()
                exc_info = None
            except:
                retval = None
                exc_info = sys.exc_info()
            result = context, retval, exc_info
            reply_method(result)


@dispatch.tasklet
def thread_pool(task_chan, max_threads=None):
    """Thread pool tasklet.

    Usage example (in a tasklet):

    # Create the thread pool:
    task_send_channel = yield dispatch.MakeChannel()
    yield dispatch.Spawn(thread_pool(task_send_channel))

    # Enqueue a task:
    result_channel = yield dispatch.MakeChannel()
    yield dispatch.Send(task_send_channel, (func, context, result_channel))
    # The Send will block until a thread is available (unless block=False is
    # given).

    # Retrieve the result:
    context, retval, exc_info = yield dispatch.Recv(result_channel)
    # The Recv will block until the result is available.
    # If func returned a value (retval), exc_info is None.
    # If func raised an exception, retval is None and exc_info is the tuple
    # (exception_type, exception_value, traceback).

    # Shut down the thread pool:
    finish_channel = yield dispatch.MakeChannel()
    yield dispatch.Send(task_send_channel, (Ellipsis, "exit", finish_channel),
                        block=False)

    # Wait until thread pool has finished:
    context, retval, exc_info = yield dispatch.Recv(finish_channel)
    # In this example, context == "exit", retval is Ellipsis, exc_info is None.

    """

    # Shuts down if func is Ellipsis, sending Ellipsis on reply_chan when all
    # threads have finished.
    if not max_threads:
        max_threads = multiprocessing.cpu_count()
    all_threads = []
    finished_threads = set() # Threads that have finished their last task.
    idle_thread_chan = yield dispatch.MakeChannel()
    idle_notify_method = yield dispatch.AsyncSender(idle_thread_chan)
    try:
        while True:
            if len(all_threads) >= max_threads:
                # Wait for an idle thread.
                thread = yield dispatch.Recv(idle_thread_chan)
            else:
                thread = None # "Create as necessary".

            func, context, reply_chan = yield dispatch.Recv(task_chan)
            if func is Ellipsis: # Sentinel.
                if thread is not None:
                    finished_threads.add(thread)
                break
            else:
                if thread is None:
                    thread = _PoolThread(idle_notify_method)
                    all_threads.append(thread)
                reply_method = yield dispatch.AsyncSender(reply_chan)
                thread.enqueue_task(func, context, reply_method)

        # Shut down working threads.
        for thread in all_threads:
            thread.finish()

        # Wait for threads to finish.
        # Threads send themselves via the idle_thread_chan after each task is
        # completed, and once more after thread.finish() is called.
        # We place threads that have finished their final task in
        # finished_threads; threads that are already in finished_threads may
        # be joined as soon as they show up in idle_thread_chan again.
        while len(all_threads):
            thread = yield dispatch.Recv(idle_thread_chan)
            if thread in finished_threads: # Thread terminated.
                thread.join()
                all_threads.remove(thread)
            else: # Thread finished last task.
                finished_threads.add(thread)

        # Signal that the thread pool has finished all tasks.
        yield Send(reply_chan, (context, Ellipsis, None))

    finally:
        # If an uncaught exception occurs, we make sure that the pool threads
        # are not leaked.
        warnings.warn("waiting for threadpool threads to finish")
        for thread in all_threads:
            thread.finish()
        for thread in all_threads:
            thread.join()

