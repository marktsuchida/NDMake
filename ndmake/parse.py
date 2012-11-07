import itertools
import re

from ndmake import pipeline
from ndmake import debug

dprint = debug.dprint_factory(__name__)
dprint_line = debug.dprint_factory(__name__, "line")
dprint_token = debug.dprint_factory(__name__, "token")
dprint_entity = debug.dprint_factory(__name__, "entity")
dprint_coroutine = debug.dprint_factory(__name__, "coroutine")


#
# Driver
#

def read_ndmakefile(file, filename=None):
    if not filename:
        try:
            filename = file.name
        except AttributeError:
            filename = None

    a_pipeline = pipeline.Pipeline()

    def parser_action(kind, name, entries):
        entity = pipeline.Entity(kind, name, entries)
        dprint_entity("entity", kind, name,
                      "entries=({})".format(", ".join(entries.keys())))
        a_pipeline.add_entity(entity)

    parser = parse(parser_action)

    def lexer_action(token):
        dprint_token("token", token)
        parser.send(token)

    lexer = lex(lexer_action)

    try:
        for i, line in enumerate(file):
            line = line.rstrip("\n")
            lineno = i + 1 # One-based line numbers.
            dprint_line("sending line to lexer", "{:d}:".format(lineno), line)
            lexer.send((line, lineno))
        lexer.close()
    except SyntaxError as e:
        e.filename = filename
        raise e

    return a_pipeline


#
# Coroutine decorator
#

# In this module, all coroutines are used as follows.
# - input is passed to coroutines via their send() method
# - output is via an action callable passed at coroutine instantiation

def coroutine(func):
    def wrapped(*args, **kwargs):
        dprint_coroutine("starting coroutine", func.__name__)
        f = func(*args, **kwargs)
        next(f)
        dprint_coroutine("started coroutine", func.__name__)
        return f
    return wrapped


def subcoroutine(func):
    # Sub-coroutines (called using `yield from') should not be primed with
    # next().
    dprint_coroutine("instantiating sub-coroutine", func.__name__)
    return func


#
# Tokens
#

class Token:
    def __init__(self, lineno, column, text, display=None):
        self.lineno = lineno
        self.column = column
        self.text = text
        self.display = display if display is not None else text

    def is_punctuation(self, punctuation):
        return isinstance(self, Punctuation) and self.text == punctuation

    def __str__(self):
        return self.display

    def __repr__(self):
        display = self.display
        if display.startswith("<") and display.endswith(">"):
            pass
        else:
            display = "`{}'".format(display)
        return "<{} {}>".format(self.__class__.name, display)

class Keyword(Token): pass
class QualifiedIdentifier(Token): pass
class Identifier(QualifiedIdentifier): pass
class Punctuation(Token): pass
class String(Token):
    def __init__(self, lineno, column, text):
        super().__init__(lineno, column, text, display="<string>")
class EndOfHeading(Token):
    def __init__(self, lineno, column):
        super().__init__(lineno, column, None, display="<eol>")
class IndentedText(Token):
    def __init__(self, lineno, column, text):
        super().__init__(lineno, column, text, display="<indented text>")
class EndOfInput(Token):
    def __init__(self, lineno, column):
        super().__init__(lineno, column, None, display="<eof>")


#
# Errors
#

class SyntaxError(Exception):
    def __init__(self, message, lineno=None, column=None, filename=None):
        if message:
            self.message = message
        self.lineno = lineno
        self.column = column
        self.filename = filename

    def __str__(self):
        filename = self.filename if self.filename else "<unknown filename>"
        if self.lineno is not None:
            if self.column is not None:
                position = "{:d}: {:d}: ".format(self.lineno, self.column)
            else:
                position = "{:d}: ".format(self.lineno)
        else:
            position = ""
        return "{}: {}".format(filename, position) + self.message


class ExpectedGotError(SyntaxError):
    def __init__(self, expected, got_token):
        self.expected = expected
        self.token = got_token
        super().__init__(None, got_token.lineno, got_token.column)

    @property
    def message(self):
        expected = (self.expected if isinstance(self.expected, str)
                    else or_join(self.expected))
        got = str(self.token)
        if got.startswith("<") and got.endswith(">"):
            got = got[1:-1]
        else:
            got = "`{}'".format(got)
        return "expected {}; got {}".format(expected, got)


def or_join(items, op="or"):
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return "{} {} {}".format(", ".join(items[:-1]), op, items[-1])


#
# Parser
#

# Usage:
# parser = parse(action)  # action is a callable accepting entity tuples
#                         # (kind, name, entries)
# lexer = lex(parser.send)

@coroutine
def parse(action):
    token = yield
    while True:
        if isinstance(token, Keyword):
            parse_entity = globals()["parse_" + token.text]
            token = yield from parse_entity(action, token)
            continue

        if isinstance(token, EndOfInput):
            return

        raise ExpectedGotError("keyword or eof", token)


def global_entity_namer():
    for i in itertools.count(0):
        yield "global_{:d}".format(i)
_global_entity_namer = global_entity_namer()
next_global_entity_name = lambda: next(_global_entity_namer)


@subcoroutine
def parse_defs(action, token):
    # "defs" {":"}? EndOfHeading {IndentedText}?
    entries = {}

    expected = "`:' or eol"
    token = yield # XXX Why is this returning None???
    if token.is_punctuation(":"):
        expected = "eol"
        token = yield
    if not isinstance(token, EndOfHeading):
        raise ExpectedGotError(expected, token)

    token = yield
    text, token = yield from parse_optional_indented_text(token)
    if text:
        entries["defs"] = text

    action("global", next_global_entity_name(), entries)
    return token


@subcoroutine
def parse_compute(action, token):
    # "compute" Identifier {"[" QualifiedIdentifier* "]"}?
    # {Identifier "=" String}* {":" Identifier*}? EndOfHeading IndentedText?
    entries = {}

    token = yield
    if not isinstance(token, Identifier):
        raise ExpectedGotError("name of computation", token)
    name = token.text

    token = yield
    if token.is_punctuation("["):
        scope, token = yield from parse_scope(token)
        entries["scope"] = scope

    try:
        pairs, token = yield from parse_pairs(("parallel",), token)
        entries.update(pairs)
    except ExpectedGotError as e:
        e.expected = list(e.expected) + ["`:'", "eol"]
        raise e

    if token.is_punctuation(":"):
        token = yield
        inputs, token = yield from parse_identifier_list(token)
        if inputs:
            entries["inputs"] = inputs

        if not isinstance(token, EndOfHeading):
            raise ExpectedGotError("input dataset name or eol", token)

    elif not isinstance(token, EndOfHeading):
        raise ExpectedGotError("keyword, `:', or eol", token)

    token = yield
    text, token = yield from parse_optional_indented_text(token)
    if text:
        entries["command"] = text

    action("compute", name, entries)
    return token


@subcoroutine
def parse_data(action, token):
    # "data" Identifier {"[" QualifiedIdentifier* "]"}? {":" Identifier?}?
    # EndOfHeading IndentedText?
    entries = {}

    token = yield
    if not isinstance(token, Identifier):
        raise ExpectedGotError("name of dataset", token)
    name = token.text

    token = yield
    if token.is_punctuation("["):
        scope, token = yield from parse_scope(token)
        entries["scope"] = scope

    # Currently, there are no key-value options for datasets.

    if token.is_punctuation(":"):
        token = yield
        if isinstance(token, Identifier):
            entries["producer"] = token.text
            token = yield

        if not isinstance(token, EndOfHeading):
            raise ExpectedGotError("producer computation name or eol", token)

    elif not isinstance(token, EndOfHeading):
        raise ExpectedGotError("`:' or eol", token)

    token = yield
    text, token = yield from parse_optional_indented_text(token)
    if text:
        entries["filename"] = text

    action("dataset", name, entries)
    return token


@subcoroutine
def parse_dimension_or_domain(action, token):
    # Keyword QualifiedIdentifier {"[" QualifiedIdentifier* "]"}? ":"?
    # EndOfHeading IndentedText
    entries = {}
    template_entry_key = token.text # values, range, or slice

    token = yield
    if not isinstance(token, QualifiedIdentifier):
        raise ExpectedGotError("dimension or subdomain name", token)
    name = token.text
    kind = "dimension" if isinstance(token, Identifier) else "subdomain"

    token = yield
    if token.is_punctuation("["):
        scope, token = yield from parse_scope(token)
        entries["given"] = scope

    expected = "`:' or eol"
    if token.is_punctuation(":"):
        expected = "eol"
        token = yield
    if not isinstance(token, EndOfHeading):
        raise ExpectedGotError(expected, token)

    token = yield
    if isinstance(token, IndentedText):
        entries[template_entry_key] = token.text
    else:
        raise ExpectedGotError("indented text", token)

    action(kind, name, entries)

    token = yield
    return token

parse_values = parse_dimension_or_domain
parse_range = parse_dimension_or_domain
parse_slice = parse_dimension_or_domain


@subcoroutine
def parse_computed_dimension_or_domain(action, token):
    # Keyword QualifiedIdentifier {"[" QualifiedIdentifier* "]"}?
    # {":" Identifier*}? EndOfHeading IndentedText
    entries = {}

    # compute_range -> range_command, etc.
    template_entry_key = token.text.split("_", 1)[1] + "_command"

    token = yield
    if not isinstance(token, QualifiedIdentifier):
        raise ExpectedGotError("dimension or subdomain name", token)
    name = token.text
    kind = "dimension" if isinstance(token, Identifier) else "subdomain"

    token = yield
    if token.is_punctuation("["):
        scope, token = yield from parse_scope(token)
        entries["given"] = scope

    # Currently, there are no key-value options.

    if token.is_punctuation(":"):
        token = yield
        inputs, token = yield from parse_identifier_list(token)
        if inputs:
            entries["inputs"] = inputs

        if not isinstance(token, EndOfHeading):
            raise ExpectedGotError("input dataset name or eol", token)

    elif not isinstance(token, EndOfHeading):
        raise ExpectedGotError("`:' or eol", token)

    token = yield
    if isinstance(token, IndentedText):
        entries[template_entry_key] = token.text
    else:
        raise ExpectedGotError("indented text", token)

    action(kind, name, entries)

    token = yield
    return token

parse_compute_values = parse_computed_dimension_or_domain
parse_compute_range = parse_computed_dimension_or_domain
parse_compute_slice = parse_computed_dimension_or_domain


@subcoroutine
def parse_data_values(action, token):
    # "data_values" QualifiedIdentifier {"[" QualifiedIdentifier* "]"}?
    # {":" Identifier?}? EndOfHeading IndentedText
    raise NotImplementedError("data_values parser not implemented yet")
    yield


#
# Parser subroutines
#

@subcoroutine
def parse_identifier_list(token):
    # Identifier*
    names = []

    while isinstance(token, Identifier):
        names.append(token.text)
        token = yield

    return names, token

@subcoroutine
def parse_scope(token):
    # "[" QualifiedIdentifier* "]"
    names = []

    token = yield
    while isinstance(token, QualifiedIdentifier):
        names.append(token.text)
        token = yield

    if not token.is_punctuation("]"):
        raise ExpectedGotError("`]'", token)

    token = yield
    return names, token


@subcoroutine
def parse_pairs(allowed_keys, token):
    # {Identifier "=" String}*
    pairs = {}

    while isinstance(token, Identifier):
        key = token.text
        if key not in allowed_keys:
            raise ExpectedGotError("keyword", token)
        if key in pairs:
            raise SyntaxError("duplicate argument key `{}'".format(key),
                              token.lineno, token.column)

        token = yield
        if not token.is_punctuation("="):
            raise ExpectedGotError("`='", token)

        token = yield
        if isinstance(token, String):
            value = token.text
        else:
            raise ExpectedGotError("quoted string", token)

        pairs[key] = value
        token = yield

    return pairs, token


@subcoroutine
def parse_optional_indented_text(token):
    if isinstance(token, IndentedText):
        text = token.text
        token = yield
        return text, token
    elif isinstance(token, Keyword) or isinstance(token, EndOfInput):
        return None, token
    else:
        raise ExpectedGotError("indented text, keyword, or eof", token)


#
# Definitions for lexer
#

hspace = " \t"
comment = "#"
qualified_word_pattern = \
        re.compile(r"[A-Za-z_][0-9A-Za-z_]*(\.[A-Za-z_][0-9A-Za-z_]*)*")
str_escapes = {
               "\n": "",
               "\\": "\\",
               "'": "'",
               '"': '"',
               "a": "\a",
               "b": "\b",
               "f": "\f",
               "n": "\n",
               "r": "\r",
               "t": "\t",
               "v": "\v",
              }
keywords = [
            "defs",
            "data",
            "compute",
            "values",
            "range",
            "slice",
            "compute_values",
            "compute_range",
            "compute_slice",
            "data_values",
           ]


#
# Lexer
#

# Usage:
# lexer = lex(action)  # action is a callable accepting Tokens
# for i, line in enumerate(lines):
#     lineno = i + 1
#     lexer.send((line, lineno))
# lexer.close()

@coroutine
def lex(action):
    non_eof_lexer = lex_non_eof(action)
    line = ""
    lineno = 0
    while True:
        try:
            line, lineno = yield
        except GeneratorExit:
            non_eof_lexer.close()
            action(EndOfInput(lineno, len(line)))
            return
        else:
            non_eof_lexer.send((line, lineno))


@coroutine
def lex_non_eof(action):
    while True:
        line, lineno = yield
        if not line.strip():
            continue # Ignore empty line.
        if line.lstrip().startswith(comment):
            continue # Ignore comment line.
        if line[0] in hspace:
            raise SyntaxError("unexpected indent", lineno=lineno)
        # Now we have the first line of the first chunk.
        break

    while True:
        heading_lexer = lex_heading(action)
        while True:
            if heading_lexer.send((line, lineno)):
                break
            line, lineno = yield
            if not line.strip():
                continue
            if line.lstrip().startswith(comment):
                continue
        
        # Now we optionally have indented lines.
        indented_lines = []
        indented_lines_lineno = None
        while True:
            try:
                line, lineno = yield
            except GeneratorExit:
                text = process_indented_lines(indented_lines)
                if text:
                    action(IndentedText(indented_lines_lineno, 0, text))
                return

            if indented_lines_lineno is None:
                indented_lines_lineno = lineno

            if not line.strip():
                indented_lines.append(line)
                continue
            if line.startswith(comment):
                continue # Ignore comment line.
            if line[0] in hspace:
                indented_lines.append(line)
                continue
            # We have found the next first line of a chunk.
            break
        text = process_indented_lines(indented_lines)
        if text:
            action(IndentedText(indented_lines_lineno, 0, text))


@coroutine
def lex_heading(action):
    # This coroutine yields a bool (True if reached end of heading line).
    line, lineno = yield
    column = 0
    bracket_level = 0
    while line or bracket_level > 0:
        if not line:
            line, lineno = yield False

        hspace_pattern = "[{}]+".format(hspace)
        m = re.match(hspace_pattern, line)
        if m:
            line = line[m.end():]
            column += m.end()
            continue

        if line[0] == "[":
            action(Punctuation(lineno, column, "["))
            bracket_level += 1
            line = line[1:]
            column += 1
            continue

        if line[0] == "]":
            action(Punctuation(lineno, column, "]"))
            bracket_level -= 1
            line = line[1:]
            column += 1
            continue

        if line[0] == "=":
            action(Punctuation(lineno, column, "="))
            line = line[1:]
            column += 1
            continue

        if line[0] == ".":
            action(Punctuation(lineno, column, "."))
            line = line[1:]
            column += 1
            continue

        if line[0] == ":":
            action(Punctuation(lineno, column, ":"))
            line = line[1:]
            column += 1
            continue

        if line[0] in ('"', "'"):
            open_quote = line[0]
            start_lineno, start_column = lineno, column
            line = line[1:]
            column += 1
            strings = []
            while True:
                string, length, finished = get_quoted_string(line, open_quote)
                strings.append(string)
                if finished:
                    line = line[length:]
                    column += length
                    break
                strings.append("\n")
                line, lineno = yield False
                column = 0
            string = "".join(strings)
            action(String(start_lineno, start_column, string))
            continue

        if line == "\\": # Backslash at end of line: continuation.
            line, lineno = yield False
            column = 0
            continue

        m = qualified_word_pattern.match(line)
        if m:
            word = m.group()
            if m.group(1): # Contains `.'
                action(QualifiedIdentifier(lineno, column, word))
            elif word in keywords:
                action(Keyword(lineno, column, word))
            else:
                action(Identifier(lineno, column, word))
            line = line[m.end():]
            column += m.end()
            continue

        # Unexpected character.
        raise SyntaxError("unexpected character ({})".format(line[0]),
                          lineno, column)

    action(EndOfHeading(lineno, column))
    yield True


def get_quoted_string(line, open_quote):
    chars = []
    it = enumerate(line)
    while True:
        pos, char = next(it)
        if char == open_quote:
            return "".join(chars), pos + 1, True
        if char == "\\":
            pos, char = next(it)
            if char in str_escapes:
                chars.append(str_escapes[char])
                continue

            if char == "x":
                digits = "".join(digit for pos, digit in
                                 (next(it) for i in range(2)))
                code = int(digits, 16)
                chars.append(chr(code))
                continue

            if char in "0123456789":
                digits = char + "".join(digit for pos, digit in
                                        (next(it) for i in range(2)))
                code = int(digits, 8)
                chars.append(chr(code))
                continue

            if char == "u":
                digits = "".join(digit for pos, digit in
                                 (next(it) for i in range(4)))
                code = int(digits, 16)
                chars.append(chr(code))
                continue

            if char == "U":
                digits = "".join(digit for pos, digit in
                                 (next(it) for i in range(8)))
                code = int(digits, 16)
                chars.append(chr(code))
                continue

            # Otherwise, not an escape.
            chars.append("\\")
            chars.append(char)
            continue

        chars.append(char)

    # We have reached the end of line without encountering the closing quote.
    return "".join(chars), pos, False


def process_indented_lines(lines):
    # Remove trailing empty lines.
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return None

    common_space_prefix = space_prefix(lines[0])
    for line in lines[1:]:
        if not line.strip():
            continue
        common_space_prefix = common_prefix(common_space_prefix,
                                            space_prefix(line))

    if not common_space_prefix:
        # Remove all leading whitespace.
        return "\n".join(line.lstrip() for line in lines)
    else:
        prefix_len = len(common_space_prefix)
        return "\n".join((line[prefix_len:] if line.strip() else "")
                         for line in lines)


def space_prefix(s):
    hspace_pattern = "[{}]*".format(hspace)
    m = re.match(hspace_pattern, s)
    return m.group()

def common_prefix(s, t):
    prefix = ""
    for c, d in zip(s, t):
        if c != d:
            break
        prefix += c
    return prefix
