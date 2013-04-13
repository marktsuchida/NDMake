import functools
import os
import os.path
import shlex
import subprocess
import sys
import urllib.parse

from ndmake import debug
from ndmake import files
from ndmake import mtime


dprint = debug.dprint_factory(__name__)
dprint_iter = debug.dprint_factory(__name__, "iter")
dprint_extent = debug.dprint_factory(__name__, "extent")
dprint_cache = debug.dprint_factory(__name__, "cache")
dprint_unlink = debug.dprint_factory(__name__, "unlink")


#
# Dimension
#

class Dimension:
    def __init__(self, name):
        self.name = name
        self.full_extent = None

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def extent_by_name(self, name):
        names = name.split(".")
        assert names[0] == self.name
        extent = self.full_extent
        for name in names[1:]:
            extent = extent.subextents[name]
        return extent


#
# Extents
#

# Implementation note: raw_sequence() is an internally used method that
# returns the sequence requested by the user. The public interface for the
# extent sequence is sequence() and iterate(). For full extents (assigned to
# dimensions), sequence() is simply raw_sequence() and iterate() iterates over
# sequence(). For subextents (assigned to subdomains), iterate() iterates over
# raw_sequence, but checks that the values in raw_sequence are indeed members
# of the superextent. Subextent sequence() is simply list(iterate()).

class Extent:
    """Representation of the value list of a dimension or subdomain."""

    def __init__(self, template=None, survey=None):
        self.template = template
        self.survey = survey

        self.name = None
        self.superextent = None
        self.subextents = {}  # name -> Extent

        self.default_format = None

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.full_name)

    @property
    def full_name(self):
        assert False, "abstract method call"

    @property
    def fullextent(self):
        return self

    @property
    def parent_vertex(self):
        if self.survey is not None:
            return self.survey
        if self.superextent is not None:
            return self.superextent.parent_vertex
        return None

    def is_demarcated(self, element):
        survey = self.parent_vertex
        if survey is None:
            return True
        return not not survey.is_result_available(element)

    def issubextent(self, other):
        if self is other:
            return True
        elif self.superextent is not None:
            if self.superextent.issubextent(other):
                return True
        return False


class SequenceExtentMixin:
    def __init__(self, *args, **kwargs):
        pass


class EnumeratedExtentMixin(SequenceExtentMixin):
    def raw_sequence(self, element):
        if self.survey is not None:
            return self.survey.result(element)

        values_str = element.render_template(self.template)
        dprint_extent(self, "values string:", values_str)
        values = shlex.split(values_str)
        dprint_extent(self, "values:", ", ".join(values))
        return values


class ArithmeticExtentMixin(SequenceExtentMixin):
    def raw_sequence(self, element):
        if self.survey is not None:
            rangeargs = self.survey.result(element)

        else:
            rangeargs_str = element.render_template(self.template)
            dprint_extent(self, "rangeargs string:", rangeargs_str)
            try:
                rangeargs = tuple(int(a) for a in rangeargs_str.split())
                assert len(rangeargs) in range(1, 4)
            except:
                raise ValueError("{}: range template must expand to 1-3 "
                                 "integers (got `{}')".
                                 format(self.full_name, rangeargs_str))

        dprint_extent(self, "rangeargs:", ", ".join(str(a) for a in rangeargs))
        return range(*rangeargs)


class FullExtent(Extent, SequenceExtentMixin):
    def __init__(self, dimension, scope, template=None, survey=None):
        super().__init__(template=template, survey=survey)
        self.dimension = dimension
        self.scope = scope
        # TODO Check that self.dimension is not in self.scope.extents

        self.value_type = (int if isinstance(self, ArithmeticExtentMixin)
                           else str)

    @property
    def full_name(self):
        return self.dimension.name

    def sequence(self, element):
        # raw_sequence() implemented by subclasses.
        return self.raw_sequence(element)

    def iterate(self, element):
        return iter(self.sequence(element))


class EnumeratedFullExtent(FullExtent, EnumeratedExtentMixin):
    pass


class ArithmeticFullExtent(FullExtent, ArithmeticExtentMixin):
    pass


class Subextent(Extent):
    def __init__(self, superextent, name, template=None, survey=None):
        super().__init__(template=template, survey=survey)
        self.name = name
        self.superextent = superextent
        self.dimension = superextent.dimension
        self.scope = superextent.scope
        self.value_type = superextent.value_type

    @property
    def full_name(self):
        return self.superextent.full_name + "." + self.name

    @property
    def fullextent(self):
        return self.superextent.fullextent


class SequenceSubextent(Subextent, SequenceExtentMixin):
    def raw_sequence(self, element):
        sequence = super().raw_sequence(element)
        if sequence and not isinstance(sequence[0], self.value_type):
            try:
                value_type = self.value_type
                sequence = list(value_type(v) for v in sequence)
            except ValueError:
                raise ("value(s) of {sub} must be members of {full} (type "
                       "mismatch)".
                       format(sub=self.full_name,
                              full=self.fullextent.name))
        return sequence

    def sequence(self, element):
        return list(self.iterate(element))

    def iterate(self, element):
        super_sequence = self.superextent.sequence(element)
        if isinstance(super_sequence, range):
            super_values = super_sequence
        else:
            super_values = frozenset(super_sequence)

        for value in self.raw_sequence(element):
            if value not in super_values:
                super_values = " ".join(str(v) for v in super_sequence)
                if not super_values:
                    super_values = "(none)"
                raise ValueError("value `{value}' generated by {sub} "
                                 "not a member of {super} "
                                 "(allowed values are: {supervalues})".
                                 format(sub=self.full_name,
                                        super=self.superextent.full_name,
                                        value=value,
                                        supervalues=super_values))
            yield value


class EnumeratedSubextent(SequenceSubextent, EnumeratedExtentMixin):
    pass


class ArithmeticSubextent(SequenceSubextent, ArithmeticExtentMixin):
    pass


class IndexedSubextent(Subextent):
    def slice(self, element):
        if self.survey is not None:
            sliceargs = self.survey.result(element)

        else:
            sliceargs_str = element.render_template(self.template)
            dprint_extent(self, "sliceargs string:", sliceargs_str)
            try:
                sliceargs = tuple(int(a) for a in sliceargs_str.split())
                assert len(sliceargs) in range(1, 4)
            except:
                raise ValueError("{}: slice template must expand to 1-3 "
                                 "integers (got `{}')".
                                 format(self.full_name, sliceargs_str))

        dprint_extent(self, "sliceargs:", ", ".join(str(a) for a in sliceargs))
        return slice(*sliceargs)

    def sequence(self, element):
        return self.superextent.sequence(element)[self.slice(element)]

    def iterate(self, element):
        return iter(self.sequence(element))


#
# Space and Element
#

class Space:
    # The __init__() implemented below can take up a significant portion of the
    # time taken to iterate over spaces, so we memoize Space instances.
    # There are more general and sophisticated methods to memoize class
    # instances, but for now we assume that Space will not be subclassed.

    _instances = {}  # tuple(manifest_extents) -> Space

    def __new__(cls, manifest_extents=[]):
        key = tuple(manifest_extents)
        if key not in cls._instances:
            cls._instances[key] = super(Space, cls).__new__(cls)
        return cls._instances[key]

    def __init__(self, manifest_extents=[]):
        if hasattr(self, "manifest_extents"):
            # Cached instance.
            return

        manifest_dims = frozenset(extent.dimension
                                  for extent in manifest_extents)
        if len(manifest_extents) > len(manifest_dims):
            raise ValueError("cannot create space with nonorthogonal "
                             "extents: {}".
                             format(", ".join(str(e)
                                              for e in manifest_extents)))
        self.manifest_extents = manifest_extents

        # Collect all the extents on which the manifest ones depend on.
        # If extents for the same dimension are encountered (between the scopes
        # of different manifest contingent extents or between the scope of
        # a manifest contingent extent and one of the manifest extents), then
        # they must be in a superextent-subextent relationship, and the
        # subextent is adopted as the extent for the whole space.
        extents_to_use = dict((extent.dimension, extent)
                              for extent in self.manifest_extents)
        for extent in self.manifest_extents:
            for extent in extent.scope.extents:
                dimension = extent.dimension
                if dimension in extents_to_use:
                    contending_extent = extents_to_use[dimension]
                    if extent.issubextent(contending_extent):
                        extents_to_use[dimension] = extent
                        continue
                    elif contending_extent.issubextent(extent):
                        continue  # Keep contending_extent for this dimension.
                    raise ValueError("cannot create space with multiple "
                                     "incompatible extents for the same "
                                     "dimension: {}".format(manifest_extents))
                else:
                    extents_to_use[dimension] = extent

        # Topologically sort the extents based on contingency relations.

        # In order to preserve the order of the manifest extents (so that the
        # user can specify the order of iteration), for each manifest extent M
        # (in order), we "claim" any remaining implicit extents {I} on which M
        # is contingent and place them to the left of M (preserving the
        # already topologically-sorted order of {I}). At the same time, we
        # make sure that the manifest extents are in fact topologically sorted.
        all_dims = set(extents_to_use.keys())
        implicit_dims = all_dims - manifest_dims
        sorted_extents = []
        checked_dimensions = set()  # Dimensions represented in sorted_extents.
        for manifest_extent in self.manifest_extents:
            if manifest_extent.dimension in checked_dimensions:
                raise ValueError("cannot create space with extents that are "
                                 "not topologically sorted")
            for e in manifest_extent.scope.extents:
                d = e.dimension
                if d in implicit_dims:
                    implicit_dims.remove(d)
                    sorted_extents.append(extents_to_use[d])
                    checked_dimensions.add(d)
            sorted_extents.append(extents_to_use[manifest_extent.dimension])
            checked_dimensions.add(manifest_extent.dimension)

        self.extents = sorted_extents
        # Some useful attributes:
        self.dimensions = list(e.dimension for e in self.extents)
        self.ndims = len(self.dimensions)

    def __repr__(self):
        return "<Space [{}]>".format(", ".join(e.full_name
                                               for e in self.extents))

    def __getitem__(self, dimension):
        for extent in self.extents:
            if extent.dimension is dimension:
                return extent
        raise KeyError("{} not in {}".format(dimension, self))

    def is_full_element(self, element):
        for extent in self.extents:
            if extent.dimension not in element.space.dimensions:
                return False
        return True

    def canonicalized_element(self, element, allow_nonsub_extents=False):
        if element.space is self:
            return element

        assigned_extents = []
        coords = {}
        element_space_dimensions = element.space.dimensions
        for extent in self.extents:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                continue
            if element.space[dimension].issubextent(extent):
                assigned_extents.append(extent)
            elif allow_nonsub_extents:
                assigned_extents.append(element.space[dimension])
            else:
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            coords[dimension] = element[dimension]
        return Element(Space(assigned_extents), coords)

    def iterate(self, element=None):
        if element is None:
            element = Element()

        # Scan _consecutive_ dimensions assigned a value by element.
        assigned_extents = []
        base_coords = {}
        element_space_dimensions = element.space.dimensions
        for extent in self.extents:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                break
            if not element.space[dimension].issubextent(extent):
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            assigned_extents.append(extent)
            base_coords[dimension] = element[dimension]

        if len(assigned_extents) == self.ndims:
            # Element assigned coordinates to all of our dimensions.
            # (We generate a canonical element directly, without having to call
            # self.canonicalized_element().)
            element = Element(self, base_coords)
            dprint_iter(self, "iterate({}): yielding full element".
                        format(element))
            yield (element, True)  # Flag indicating full element.
            return

        # We will iterate over the first unassigned extent.
        extent_to_iterate = self.extents[len(assigned_extents)]
        assigned_extents.append(extent_to_iterate)

        # Scan remaining (nonconsecutive) dimensions assigned by element.
        for extent in self.extents[len(assigned_extents):]:
            dimension = extent.dimension
            if dimension not in element_space_dimensions:
                continue
            if not element.space[dimension].issubextent(extent):
                raise ValueError("{} of {} is not subextent of {} of {}".
                                 format(element.space[dimension],
                                        element, extent, self))
            assigned_extents.append(extent)
            base_coords[dimension] = element[dimension]

        if extent_to_iterate.is_demarcated(element):
            dprint_iter(self, "iterate({}): iterating extent:".format(element),
                        extent_to_iterate)
            for value in (extent_to_iterate.iterate(element)):
                base_coords[extent_to_iterate.dimension] = value
                new_element = Element(Space(assigned_extents), base_coords)
                dprint_iter(self, "iterate({}): descending into:".
                            format(element), new_element)
                yield from self.iterate(new_element)
            dprint_iter(self, "iterate({}): finished".format(element))
        else:  # Undemarcated: yield a partial element.
            assigned_extents.remove(extent_to_iterate)
            new_element = Element(Space(assigned_extents), base_coords)
            dprint_iter(self, "iterate({}) yielding partial element:".
                        format(element), new_element)
            yield (new_element, False)  # Flag indicating partial element.


class Element:
    def __init__(self, space=Space(), coordinates=dict()):
        self.space = space
        self.coordinates = {}  # dim -> value
        for extent in self.space.extents:
            dim = extent.dimension
            if dim in coordinates:
                self.coordinates[dim] = coordinates[dim]
                continue
            assert False, ("attempt to construct incomplete Element for {}: "
                           "coordinates: {}".format(space, coordinates))

    def __len__(self):
        return self.space.ndims

    def __getitem__(self, dimension):
        return self.coordinates[dimension]

    def __hash__(self):
        return hash(tuple(self.coordinates[dim]
                          for dim in self.space.dimensions))

    def __eq__(self, other):
        self_tuple = tuple(self.coordinates[dim]
                           for dim in self.space.dimensions)
        other_tuple = tuple(other.coordinates[dim]
                            for dim in other.space.dimensions)
        return self_tuple == other_tuple

    def __repr__(self):
        coords = ("{}={}".format(extent.full_name, self[extent.dimension])
                  for extent in self.space.extents)
        return "<Element [{}]>".format(", ".join(coords))

    def render_template(self, template, *, extra_names={}):
        dict_ = dict((dim.name,
                      (("{:" + dim.default_format + "}").format(self[dim])
                       if dim.default_format else self[dim]))
                     for dim in self.space.dimensions)
        dict_.update(extra_names)
        dict_["__path__"] = files.element_path(self)
        rendition = template.render(dict_)
        return rendition

    def appended(self, extent, value):
        space = Space(self.space.extents + [extent])
        coords = self.coordinates.copy()
        coords[extent.dimension] = value
        return Element(space, coords)

    def inserted(self, index, extent, value):
        space = Space(self.space.extents[:index] + [extent] +
                      self.space.extents[index:])
        coords = self.coordinates.copy()
        coords[extent.dimension] = value
        return Element(space, coords)


#
# Surveyer
#

class Surveyer:
    mtimes_include_files = False

    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    def read_mtimes(self, element):
        filename = self.cache_file(element)
        return mtime.get(filename)

    def load_result(self, element):
        with open(self.cache_file(element)) as file:
            result_text = file.read()
        return result_text

    def delete_files(self, element, delete_surveyed_files=False):
        assert not delete_surveyed_files
        filename = self.cache_file(element)
        dprint_unlink("deleting", filename)
        if os.path.exists(filename):
            os.unlink(filename)

    def create_dirs(self, element):
        pass  # No-op.


class CommandSurveyer(Surveyer):
    def __init__(self, name, scope, command_template):
        super().__init__(name, scope)
        self.command_template = command_template

    def cache_file(self, element):
        return os.path.join(files.ndmake_dir(), "survey", self.name,
                            files.element_path(element), "output")

    def load_result(self, element):
        result_text = super().load_result(element)
        result = self.convert_result(result_text)
        return result

    def run_survey(self, survey, element, extra_names, options):
        command = element.render_template(self.command_template,
                                          extra_names=extra_names)

        if options.get("print_executed_commands", False):
            print(command)
        # TODO Command should probably run in thread.
        with subprocess.Popen(command, shell=True,
                              bufsize=-1, universal_newlines=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as proc:
            result_text, error_text = proc.communicate()
            if error_text:
                print(error_text, file=sys.stderr)
            if proc.returncode:
                raise CalledProcessError("command returned exit status of "
                                         "{:d}: {}".format(proc.returncode,
                                                           command))

        cache_file = self.cache_file(element)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as file:
            file.write(result_text)

        result = self.convert_result(result_text)
        return result


class IntegerTripletCommandSurveyer(CommandSurveyer):
    def convert_result(self, result_text):
        try:
            result = tuple(int(v) for v in result_text.split())
            assert len(result) in range(1, 4)
        except:
            raise ValueError("command output does not conform to range or "
                             "slice argument format (1-3 integers required)")
        return result


class ValuesCommandSurveyer(CommandSurveyer):
    def __init__(self, name, scope, command_template, transform_template=None):
        if transform_template is not None:
            raise NotImplementedError("transform template not implemented")
        super().__init__(name, scope, command_template)

    def convert_result(self, result_text):
        return result_text.splitlines()


class FilenameSurveyer(Surveyer):
    mtimes_include_files = True

    def __init__(self, name, scope, pattern_template, transform_template=None):
        if transform_template is not None:
            raise NotImplementedError("transform template not implemented")
        super().__init__(name, scope)
        self.pattern_template = pattern_template

    def cache_file(self, element):
        return os.path.join(files.ndmake_dir(), "survey", self.name,
                            files.element_path(element), "roster")

    def read_mtimes(self, element):
        roster_old, roster_new = super().read_mtimes(element)
        if mtime.missing(roster_old, roster_new):
            return mtime.MISSING

        roster_text = super().load_result(element)
        our_oldest, our_newest = roster_old, roster_new
        for filename, recorded_mtime in (shlex.split(line)[:2]
                                         for line in roster_text.splitlines()):
            file_oldest, file_newest = mtime.get(filename)
            if mtime.missing(file_oldest, file_newest):
                return mtime.MISSING
            assert file_oldest == file_newest
            if file_oldest != recorded_mtime:
                return mtime.MISSING
            our_oldest = min(our_oldest, file_oldest)
            our_newest = max(our_newest, file_oldest)
        return our_oldest, our_newest

    def load_result(self, element):
        roster_text = super().load_result(element)
        # TODO Do pattern matching and transformation on each filename and
        # return list of results. Or load saved transformed results.
        raise NotImplementedError()

    def run_survey(self, survey, element, extra_names, options):
        # TODO Do pattern matching in a smart way (dir-by-dir)
        # save file list (with transformed result?)
        # return list of transformed results.
        raise NotImplementedError()

    def delete_files(self, element, delete_surveyed_files):
        if delete_surveyed_files:
            try:
                roster_text = super().load_result(element)
            except FileNotFoundError:
                pass
            else:
                for filename in (shlex.split(line)[0]
                                 for line in roster_text.splitlines()):
                    dprint_unlink("deleting", filename)
                    if os.path.exists(filename):
                        os.unlink(filename)
        super().delete_files(element)

    def create_dirs(self, element):
        # TODO We need the associated dataset for this...
        # Then, just run dataset.create_dirs(element)
        raise NotImplementedError()


#
# Spatial cache
#

class Cache:
    # A cache to avoid repeated retrieval and computation of aggregate
    # statistics for subspaces. The "aggregate statistics" are usually the
    # oldest and newest mtime, but this class is designed to be general. The
    # cache holds values for all full elements of the associated space, as well
    # as aggregate values (ultimately computed from the values for the full
    # elements) for each left-contiguous partial elements. Values for
    # non-contiguous elements can be computed but are not cached.
    #
    # Upon initialization, the Cache is given a loader function, which, given a
    # full element, returns the corresponding value. The combiner function,
    # also specified on initialization, takes an iterator that yields values
    # and returns a single aggregate value. Both values retrieved by the loader
    # and values computed by the combiner are cached, so that the loading and
    # aggregation is repeated as infrequently as possible.
    #
    # When the value for an element (full or partial) is invalidated (deleted),
    # values for all partial elements containing that element are also cleared.
    # This ensures that all values that might have changed are cleared. When a
    # partial element is invalidated, any sub-elements are also removed.
    #
    # The Cache object can be saved to a file (see the set_persistence()
    # method). The file used for cache persistence is automatically deleted
    # when it is determined to be outdated. Once persistence is enabled, the
    # file is loaded automatically and lazily.
    #
    # Only up to a set "level" is saved to the persistence file. In other
    # words, what is persisted can be limited to partial elements of a given
    # dimensionality. When such a file is loaded, the cache is in a state where
    # values are "cached" for partial elements even though values for their
    # full elements are not cached. Cache misses for deeper elements will cause
    # the corresponding values to be cached, but will not invalidate the loaded
    # values for partial elements (which is correct, under the assumption that
    # the cache remains valid).

    # XXX TODO Deal with undemarcated extents. Or no need?

    def __init__(self, space, loader, combiner, _level=0):
        # loader   - callable providing the uncached value for a full element
        # combiner - callable taking iterator as argument and returning the
        #            aggregate value
        self.space = space
        self.level = _level  # Index of extent within self.space.
        if self.level < self.space.ndims:
            self.extent = self.space.extents[self.level]
        self.load = loader
        self.combine = combiner

        self.cached = None  # The cached value.
        self.map = {}  # 1st-D key -> Cache for next level

        self.has_loaded_from_file = False
        self.has_deleted_file = False

    def set_persistence(self, reader, writer, path, filename, level):
        # The reader and writer are used to (de)serialize values.
        # Only the toplevel Cache uses these attributes.
        self.reader = reader
        self.writer = writer
        self.persistence_filename = os.path.join(path, filename)
        self.persistence_level = level

    def __getitem__(self, element):
        # Used only on the toplevel Cache.
        self._load_from_file()
        element = self.space.canonicalized_element(element)
        return self._get(element, 0, Element())

    def __delitem__(self, element):
        # Used only on the toplevel Cache.
        self._load_from_file()
        element = self.space.canonicalized_element(element,
                                                   allow_nonsub_extents=True)
        self._invalidate(element)
        self._delete_file()

    def _subtree(self, key):
        if key not in self.map:
            self.map[key] = Cache(self.space, self.load, self.combine,
                                  _level=self.level + 1)
        return self.map[key]

    def _get(self, key_element, dim_index, full_element):
        # full_element always ends one dimension above.

        if not (len(key_element) - dim_index):
            # Reached end of key_element.
            return self._compute(full_element)

        if self.extent.dimension in key_element.space.dimensions:
            # Key element assigns our dimension.
            key = key_element[self.extent.dimension]
            return self._subtree(key)._get(key_element, dim_index + 1,
                                           full_element.
                                           appended(self.extent, key))

        # Key element skips our dimension. Compute on the fly, but don't cache
        # at our level.
        return self.combine(self._subtree(key).
                            _get(key_element, dim_index,
                                 full_element.appended(self.extent, key))
                            for key in self.extent.iterate(key_element))

    def _compute(self, full_element):
        if self.cached is not None:
            pass  # Use cached.

        elif not (self.space.ndims - self.level):
            # We are a leaf.
            self.cached = self.load(full_element)

        else:
            # We are not a leaf.
            self.cached = self.combine(self._subtree(key).
                                       _compute(full_element.
                                                appended(self.extent, key))
                                       for key
                                       in self.extent.iterate(full_element))

        return self.cached

    def _invalidate(self, key_element, dim_index=0):
        self.cached = None

        if not (len(key_element) - dim_index):
            # Reached end of key_element.
            self.map.clear()
            return

        if self.extent.dimension in key_element.space.dimensions:
            # Key element assigns our dimension.
            key = key_element[self.extent.dimension]
            self._subtree(key)._invalidate(key_element, dim_index + 1)
            return

        # Key element skips our dimension.
        for key in self.extent.iterate(key_element):
            new_key_element = key_element.inserted(dim_index, self.extent, key)
            self._subtree(key)._invalidate(new_key_element, dim_index + 1)

    def _set(self, element, value, dim_index=0):
        if not (len(element) - dim_index):
            self.cached = value
            return

        key = element[self.extent.dimension]
        self._subtree(key)._set(element, value, dim_index + 1)

    def _write_file_lines(self, writer, fprint, element, level):
        if self.cached is not None:
            fprint(shlex.quote(os.path.join("/", files.element_path(element))),
                   shlex.quote(writer(self.cached)))
        if len(element) == self.space.ndims or not level:
            return
        for key, subtree in self.map.items():
            subtree._write_file_lines(writer, fprint,
                                      element.appended(self.extent, key),
                                      level - 1)

    def _load_from_file(self):
        if self.has_loaded_from_file or self.has_deleted_file:
            return
        if not hasattr(self, "persistence_filename"):
            return
        if not os.path.exists(self.persistence_filename):
            self.has_deleted_file = True
            return

        dprint_cache("Cache._load_from_file", self.persistence_filename)
        with open(self.persistence_filename) as file:
            try:
                for line in file:
                    element_path, value_string = shlex.split(line)
                    element_items = element_path.split("/")
                    assert not element_items[0]
                    if not element_items[1]:
                        element_items.pop()
                    coords = {}
                    for i, component in enumerate(element_items[1:]):
                        dim = self.space.dimensions[i]
                        dim_name, escaped_key = component.split("=", 1)
                        assert dim_name == dim.name
                        key = urllib.parse.unquote(escaped_key)
                        coords[dim] = dim.full_extent.value_type(key)
                    element = Element(Space(self.space.extents[:len(coords)]),
                                      coords)
                    value = self.reader(value_string)
                    self._set(element, value)
            except:
                self.cached = None
                self.map.clear()
                dprint_cache("Cache: cannot read file; deleting:",
                             self.persistence_filename)
                self._delete_file()
                return

        self.has_loaded_from_file = True

    def save_to_file(self):
        if not hasattr(self, "persistence_filename"):
            return

        os.makedirs(os.path.dirname(self.persistence_filename), exist_ok=True)
        dprint_cache("Cache.save_to_file", self.persistence_filename)
        with open(self.persistence_filename, "w") as file:
            fprint = functools.partial(print, file=file)
            self._write_file_lines(self.writer, fprint,
                                   Element(), self.persistence_level)

        self.has_deleted_file = False

    def _delete_file(self):
        if self.has_deleted_file:
            return
        if not hasattr(self, "persistence_filename"):
            return
        dprint_unlink("deleting", self.persistence_filename)
        if os.path.exists(self.persistence_filename):
            dprint_cache("Cache._delete_file", self.persistence_filename)
            os.unlink(self.persistence_filename)
        self.has_deleted_file = True
