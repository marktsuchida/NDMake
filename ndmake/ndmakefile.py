import configparser
import functools

from ndmake import pipeline
from ndmake import debug

dprint = debug.dprint_factory(__name__)

# Parse an ndmake input file (an ndmakefile).
#
# The sections and entries of the configparser format (hybrid INI/RFC 822
# format) are first transformed into a graph of sections (Section objects) that
# describes the declarative dependencies between them (so that they can appear
# in arbitrary order in the input file). The section graph is topologically
# sorted before being transformed into a dependency graph (depgraph.Graph and
# associated objects), so that we do not need to introduce placeholders in the
# depgraph.


class Error(Exception):
    def __init__(self, message, section_kind=None, section_name=None):
        super().__init__(message, section_kind, section_name)

    @property
    def message(self): return self.args[0]

    @property
    def kind(self): return self.args[1]

    @property
    def name(self): return self.args[2]

    def __str__(self):
        if self.kind is not None:
            if self.name is not None:
                return "{} {}: {}".format(self.kind, self.name, self.message)
            return "{}: {}".format(self.kind, self.message)
        return self.message


class NDMakefile:
    def __init__(self, file):
        parser = configparser.ConfigParser(allow_no_value=False,
                                           delimiters=(":",),
                                           comment_prefixes=("#",),
                                           strict=True, # Disallow duplicates.
                                           empty_lines_in_values=True,
                                           default_section="global",
                                           interpolation=None)
        parser.read_file(file)
        self.parser = parser

    def pipeline(self):
        pline = pipeline.Pipeline()
        for entity in self.entities():
            dprint("parsed", entity)
            try:
                pline.add_entity(entity)
            except Exception as e:
                raise Error(str(e), entity.kind, entity.name)
        return pline

    def entities(self):
        for section in self.parser.sections():
            words = section.split()
            if not len(words):
                raise Error("missing section title")
            if len(words) > 2:
                raise Error("too many words in section title: {}".
                            format(" ".join(words)))

            kind = words[0]
            name = words[1] if len(words) == 2 else None
            entries = dict(self.parser.items(section))
            try:
                processor = getattr(self, "process_{}_entity".format(kind))
            except KeyError:
                raise Error("unknown section type: {}".format(kind))
            processor(entries)

            yield pipeline.Entity(kind, name, entries)

    def process_global_entity(self, entries):
        pass

    def process_dimension_entity(self, entries):
        if "given" in entries:
            entries["given"] = entries["given"].split()
        if "inputs" in entries:
            entries["inputs"] = entries["inputs"].split()

    def process_subdomain_entity(self, entries):
        if "inputs" in entries:
            entries["inputs"] = entries["inputs"].split()

    def process_dataset_entity(self, entries):
        if "scope" in entries:
            entries["scope"] = entries["scope"].split()

    def process_compute_entity(self, entries):
        if "scope" in entries:
            entries["scope"] = entries["scope"].split()
        if "inputs" in entries:
            entries["inputs"] = entries["inputs"].split()

