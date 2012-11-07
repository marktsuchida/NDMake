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
    # We cannot just escape values for the shell (shlex.quote), because slashes
    # are not allowed in directory names. So we use URL-style %-encoding, but
    # keep some common shell-safe characters unescaped.
    # This will %-encode all characters except for alphanumerics and the
    # specified "safe" characters, which are safe to have in filenames.
    return urllib.parse.quote(str(value), safe=" +,-.=@_")


def element_path(element):
    dirs = list("{}={}".
                format(extent.dimension.name,
                       escape_value_for_filename(element[extent.dimension]))
                for extent in element.space.extents)
    if not dirs:
        return ""
    return os.path.join(*dirs)

