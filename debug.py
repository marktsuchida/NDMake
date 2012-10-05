
DEBUG = True

def dprint_factory(module_name, enabled):
    if DEBUG and enabled:
        def dprint(*args):
            print("{}: {}:".format(module_name, args[0]),
                  *args[1:])
    else:
        def dprint(*args): pass
    return dprint

