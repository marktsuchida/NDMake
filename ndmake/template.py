import jinja2, jinja2.meta, jinja2.sandbox
import re
import shlex

from ndmake import debug

dprint = debug.dprint_factory(__name__, False)

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
        self.sources = {"__globaldefs": ""}

        def load_template(template_name):
            prefix = ("{% import '__globaldefs' as g %}"
                      if template_name != "__globaldefs" else "")
            source = self.sources[template_name]
            return prefix + source
        loader = jinja2.FunctionLoader(load_template)
        e = jinja2.sandbox.ImmutableSandboxedEnvironment(loader=loader)

        import os.path
        e.filters["dirname"] = os.path.dirname
        e.filters["basename"] = os.path.basename
        e.filters["stripext"] = lambda p: os.path.splitext(p)[0]
        e.filters["fileext"] = lambda p: os.path.splitext(p)[1]
        e.filters["shquote"] = shellquote
        e.filters["shsplit"] = lambda l: shlex.split(str(l))

        self.environment = e

    def new_template(self, name, source):
        return Template(self, name, source)

    def append_global_defs(self, source):
        self.sources["__globaldefs"] = self.sources["__globaldefs"] + source


class Template:

    def __init__(self, templateset, name, source):
        templateset.sources[name] = source
        # dprint("template registered", name)
        self.templateset = templateset
        self.name = name
        self.source = source

        self.params = None

    @property
    def environment(self):
        return self.templateset.environment

    @property
    def parameters(self):
        if self.params is None:
            ast = self.environment.parse(self.source, self.name)
            self.params = jinja2.meta.find_undeclared_variables(ast)
        return self.params

    def render(self, dict_):
        # Check that all variables are defined.
        params = self.parameters
        for param in params:
            if param not in dict_ and param != "g":
                dprint("rendering " + self.name,
                       "source: \"\"\"{}\"\"\",".format(self.source),
                       "dict:", str(dict_))
                raise KeyError("undefined template variable: {}".format(param))

        tmpl = self.environment.get_template(self.name)
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


_before_unsafe_char = re.compile("(?=[^-_./A-Za-z0-9])")
def shellquote(name):
    if name:
        return (_before_unsafe_char.sub(r"\\", name). # Prefix unsafe with '\'.
                replace("\\\n", "'\n'")) # Handle newline specially.
    return "''"

