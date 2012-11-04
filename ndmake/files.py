import urllib.parse
import os, os.path

#
# File handling utility functions
#

def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        os.utime(path)
    return os.path.getmtime(path)


def ndmake_dir():
    return "__ndmake__"


def escape_value_for_filename(value):
    # This will %-encode all characters except for alphanumerics and the
    # specified "safe" characters, which are safe to have in filenames.
    return urllib.parse.quote(value, safe=" +,-.=@_")


def element_dirs(element):
    dirs = list("{}={}".
                format(extent.dimension.name,
                       escape_value_for_filename(element[extent.dimension]))
                for extent in element.space.extents)
    if not dirs:
        return ""
    return os.path.join(*dirs)

