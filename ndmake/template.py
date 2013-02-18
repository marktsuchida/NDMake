import jinja2, jinja2.meta, jinja2.sandbox
import re
import shlex

from ndmake import debug

dprint = debug.dprint_factory(__name__)
dprint_globals = debug.dprint_factory(__name__, "globals")

class Environment:
    def __init__(self):
        self._jinja2_environment = None # Create lazily.
        self.sources = {} # template name -> source text
        self.globals = {} # name -> value

    @property
    def jinja2_environment(self):
        if not self._jinja2_environment:
            loader = jinja2.FunctionLoader(lambda name: self.sources[name])
            e = jinja2.sandbox.ImmutableSandboxedEnvironment(loader=loader)

            import os.path
            e.filters["dirname"] = os.path.dirname
            e.filters["basename"] = os.path.basename
            e.filters["stripext"] = lambda p: os.path.splitext(p)[0]
            e.filters["fileext"] = lambda p: os.path.splitext(p)[1]
            e.filters["shquote"] = lambda s: shlex.quote(str(s))
            e.filters["shsplit"] = lambda l: shlex.split(str(l))

            e.globals.update(self.globals)

            self._jinja2_environment = e
        return self._jinja2_environment

    def parse(self, source, name):
        return self.jinja2_environment.parse(source, name)

    def get_template(self, name):
        return self.jinja2_environment.get_template(name)

    def new_template(self, name, source):
        return Template(self, name, source)

    def append_global_defs(self, source):
        # Extract the names defined in source and add them to self.globals,
        # which will be added to the environment.
        jinja2_tmpl = self.jinja2_environment.from_string(source)
        tmpl_module = jinja2_tmpl.module
        for name in dir(tmpl_module):
            if not name.startswith("_"):
                self.globals[name] = getattr(tmpl_module, name)
                dprint_globals("added global", name, self.globals[name])
        # Invalidate the existing environment.
        self._jinja2_environment = None


class Template:

    def __init__(self, environment, name, source):
        environment.sources[name] = source
        # dprint("template registered", name)
        self.environment = environment
        self.name = name
        self.source = source

        self.params = None

    @property
    def parameters(self):
        if self.params is None:
            ast = self.environment.parse(self.source, self.name)
            self.params = jinja2.meta.find_undeclared_variables(ast)
        return self.params

    def render(self, dict_):
        # Check that all variables are defined (to simplify error messages).
        params = self.parameters
        for param in params:
            globals = self.environment.jinja2_environment.globals
            if param not in dict_ and param not in globals:
                dprint("rendering " + self.name,
                       "source: {},".format(repr(self.source)),
                       "dict:", str(dict_))
                dprint_globals("globals", self.environment.globals)
                raise KeyError("undefined template variable: {}".format(param))

        jinja2_tmpl = self.environment.get_template(self.name)
        try:
            rendition = jinja2_tmpl.render(dict_)
        except Exception as e:
            raise ValueError("template {t} (source: {s}, "
                             "dict: {d}) could not be rendered: {e}".
                             format(t=self.name, s=repr(self.source),
                                    d=dict_, e=e))

        dprint("rendered " + self.name,
               "source: {},".format(repr(self.source)),
               "dict:", str(dict_) + ",",
               "result: {}".format(repr(rendition)))
        return rendition

