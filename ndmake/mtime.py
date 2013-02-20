import os.path
import time

from ndmake import debug


_dprint_mtime = debug.dprint_factory(__name__, "mtime")


def strfmtime(mtime):
    return (time.strftime("%Y%m%dT%H%M%S", time.localtime(mtime)) +
            ".{:04d}".format(round(mtime % 1 * 10000)))


def dprint_mtime(mtime, *args):
    # Wrapper to avoid strftime() call when not printing.
    if debug.dprint_enabled.get(__name__ + "_mtime"):
        if isinstance(mtime, int) or isinstance(mtime, float):
            _dprint_mtime("mtime", strfmtime(mtime), *args)
        else:
            _dprint_mtime("mtime", mtime, *args)


# FAR_PAST and FAR_FUTURE: constants representing missing time.
# They are made to be "valid" time-since-epoch values, so that calling ctime()
# on them does not raise errors.

FAR_PAST = -2**64
while True:
    try:
        time.asctime(time.gmtime(FAR_PAST))
        time.ctime(FAR_PAST)
    except:
        FAR_PAST >>= 1
    else:
        # Clip to a recognizable minimum to aid debugging.
        min_far_past = time.mktime((1, 1, 1, 0, 0, 0, 0, 0, 0))
        FAR_PAST = max(FAR_PAST, min_far_past)
        break

FAR_FUTURE = 2**64 - 1
while True:
    try:
        time.asctime(time.gmtime(FAR_FUTURE))
        time.ctime(FAR_FUTURE)
    except:
        FAR_FUTURE >>= 1
    else:
        # Clip to a recognizable maximum to aid debugging.
        max_far_future = time.mktime((2999, 12, 31, 23, 59, 59, 0, 0, 0))
        FAR_FUTURE = min(FAR_FUTURE, max_far_future)
        break


def get(filename):
    try:
        mtime = os.path.getmtime(filename)
        dprint_mtime(mtime, filename)
    except FileNotFoundError:
        dprint_mtime("missing", filename)
        return FAR_PAST, FAR_FUTURE
    return mtime, mtime  # oldest, newest


def extrema(iter):
    # For use as combiner for space.Cache.
    oldest, newest = FAR_FUTURE, FAR_PAST
    for old, new in iter:
        if old == FAR_PAST or new == FAR_FUTURE:
            assert (old, new) == (FAR_PAST, FAR_FUTURE)
            return FAR_PAST, FAR_FUTURE
        oldest = min(oldest, old)
        newest = max(newest, new)
    if (oldest, newest) == (FAR_FUTURE, FAR_PAST):
        # iter did not yield anything.
        return FAR_PAST, FAR_FUTURE
    return oldest, newest


def reader(s):
    # For use as reader for space.Cache persistence.
    return tuple(float(t) for t in s.split())


def writer(mtimes):
    # For use as writer for space.Cache persistence.
    return "{:f} {:f}".format(*mtimes)
