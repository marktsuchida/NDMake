import jinja2, jinja2.meta, jinja2.sandbox


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

        self.filename_expander = None
        @jinja2.contextfunction
        def filename(context, dataset_name, **extra_coords):
            if self.filename_expander is None:
                raise ValueError("filename function not allowed in this "
                                 "template")
            return self.filename_expander(dataset_name, extra_coords)
        e.globals["filename"] = filename

        self.environment = e

    def new_template(self, name, source):
        return Template(self, name, source)

    def append_global_defs(self, source):
        self.sources["__globaldefs"] = self.sources["__globaldefs"] + source


class Template:

    def __init__(self, templateset, name, source):
        templateset.sources[name] = source
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

    def render(self, dict_, filename_expander=None):
        save_expander = self.templateset.filename_expander
        self.templateset.filename_expander = filename_expander

        tmpl_impl = self.environment.get_template(self.name)
        rendition = tmpl_impl.render(dict_)

        self.templateset.filename_expander = save_expander
        return rendition

