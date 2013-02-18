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


# A constant representing a "valid" time-since-epoch value.
# This is only done to prevent errors when printing ctime(MAX_TIME).
MAX_TIME = 2**64 - 1
while True:
    try:
        time.asctime(time.gmtime(MAX_TIME))
        time.ctime(MAX_TIME)
    except:
        MAX_TIME >>= 1
    else:
        # Clip to a recognizable maximum to aid debugging.
        max_max_time = time.mktime((2999, 12, 31, 23, 59, 59, 0, 0, 0))
        MAX_TIME = min(MAX_TIME, max_max_time)
        break


def get(filename):
    try:
        mtime = os.path.getmtime(filename)
        dprint_mtime(mtime, filename)
    except FileNotFoundError:
        dprint_mtime("missing", filename)
        return 0, MAX_TIME
    return mtime, mtime  # oldest, newest


def extrema(iter):
    # For use as combiner for space.Cache.
    oldest, newest = MAX_TIME, 0
    for old, new in iter:
        if old == 0 or new == MAX_TIME:
            assert (old, new) == (0, MAX_TIME)
            return 0, MAX_TIME
        oldest = min(oldest, old)
        newest = max(newest, new)
    if (oldest, newest) == (MAX_TIME, 0):
        # iter did not yield anything.
        return 0, MAX_TIME
    return oldest, newest


def reader(s):
    # For use as reader for space.Cache persistence.
    return tuple(float(t) for t in s.split())


def writer(mtimes):
    # For use as writer for space.Cache persistence.
    return "{:f} {:f}".format(*mtimes)
