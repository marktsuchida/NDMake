import jinja2, jinja2.meta, jinja2.sandbox
import re
import shlex

from ndmake import debug

dprint = debug.dprint_factory(__name__)

# Notes.
# - There was the idea of a list-template. It might be possible to list-ify
#   string templates (before rendering) using the template AST. However, quote
#   handling is complex and probably not worthwhile at this stage.
# - It is also possible that the desired speedup could be obtained by applying
#   shlex after template rendering, thereby bypassing the shell.
# - In any case, knowledge that the command does not use shell syntax is
#   necessary.

class TemplateSet:
    def __init__(self):
        self._environment = None # Create lazily.
        self.sources = {} # template name -> source text
        self.globals = {} # name -> value

    @property
    def environment(self):
        if not self._environment:
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

            self._environment = e
        return self._environment

    def new_template(self, name, source):
        return Template(self, name, source)

    def append_global_defs(self, source):
        # Extract the names defined in source and add them to self.globals,
        # which will be added to the environment.
        tmpl = self.environment.from_string(source)
        tmpl_module = tmpl.module
        for name in dir(tmpl_module):
            if not name.startswith("_"):
                self.globals[name] = getattr(tmpl_module, name)
        # Invalidate the existing environment.
        self._environment = None


class Template:

    def __init__(self, templateset, name, source):
        templateset.sources[name] = source
        # dprint("template registered", name)
        self.templateset = templateset
        self.name = name
        self.source = source

        self.params = None

    @property
    def parameters(self):
        if self.params is None:
            ast = self.templateset.environment.parse(self.source, self.name)
            self.params = jinja2.meta.find_undeclared_variables(ast)
        return self.params

    def render(self, dict_):
        # Check that all variables are defined (to simplify error messages).
        params = self.parameters
        for param in params:
            if param not in dict_ and param not in self.templateset.globals:
                dprint("rendering " + self.name,
                       "source: \"\"\"{}\"\"\",".format(self.source),
                       "dict:", str(dict_))
                raise KeyError("undefined template variable: {}".format(param))

        tmpl = self.templateset.environment.get_template(self.name)
        try:
            rendition = tmpl.render(dict_)
        except Exception as e:
            raise ValueError("template {t} (source: \"\"\"{s}\"\"\", "
                             "dict: {d}) could not be rendered: {e}".
                             format(t=self.name, s=self.source,
                                    d=dict_, e=e))

        dprint("rendered " + self.name,
               "source: \"\"\"{}\"\"\",".format(self.source),
               "dict:", str(dict_) + ",",
               "result: \"\"\"{}\"\"\"".format(rendition))
        return rendition

