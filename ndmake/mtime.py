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
        try:
            # This overflows on some OSs (e.g. OS X).
            min_far_past = time.mktime((1000, 1, 1, 0, 0, 0, 0, 0, 0))
        except:
            min_far_past = 0
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


# In general, we handle mtimes as the pair (oldest, newest). For a single file,
# oldest == newest == file's mtime. When dealing with a set of files, the two
# items indicate the mtime extrema.

# The get() function below reports MISSING == (FAR_PAST, FAR_FUTURE) for files
# that don't exist. This gets propagated to file set mtimes via extrema().

MISSING = (FAR_PAST, FAR_FUTURE)
_EMPTY_EXTREMA = (FAR_FUTURE, FAR_PAST)


def get(filename):
    try:
        mtime = os.path.getmtime(filename)
        dprint_mtime(mtime, filename)
    except FileNotFoundError:
        dprint_mtime("missing", filename)
        return MISSING
    return mtime, mtime  # oldest, newest


def missing(oldest, newest):
    return oldest == FAR_PAST or newest == FAR_FUTURE


def extrema(iter):
    # For use as combiner for space.Cache.
    # iter yields (old, new) pairs.
    oldest, newest = _EMPTY_EXTREMA
    for old, new in iter:
        if missing(old, new):
            assert (old, new) == MISSING
            return MISSING
        oldest = min(oldest, old)
        newest = max(newest, new)
    if (oldest, newest) == _EMPTY_EXTREMA:
        # iter did not yield anything.
        return MISSING
    return oldest, newest


def reader(s):
    # For use as reader for space.Cache persistence.
    return tuple(float(t) for t in s.split())


def writer(mtimes):
    # For use as writer for space.Cache persistence.
    return "{:f} {:f}".format(*mtimes)
