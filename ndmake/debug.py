import warnings

DEBUG = True  # Must be set at module import time.

dprint_enabled = {}


def enable_dprint(category, enable=True):
    if category not in dprint_enabled:
        warnings.warn("unknown debug category: {}".format(category))
    dprint_enabled[category] = enable
    print("debug: enabling", category)


def categories():
    return sorted(dprint_enabled.keys())


def dprint_factory(module_name, subcategory=None):
    module_name = module_name.split(".")[-1]
    category = ("_".join((module_name, subcategory))
                if subcategory else module_name)
    dprint_enabled[category] = False
    if DEBUG:
        def dprint(*args):
            if dprint_enabled[category]:
                print("{}: {}:".format(module_name, args[0]),
                      *args[1:])
    else:
        def dprint(*args):
            pass
    return dprint
