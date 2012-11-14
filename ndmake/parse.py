import itertools
import re

from ndmake import pipeline
from ndmake import debug

dprint = debug.dprint_factory(__name__)
dprint_line = debug.dprint_factory(__name__, "line")
dprint_string = debug.dprint_factory(__name__, "string")
dprint_expression = debug.dprint_factory(__name__, "expression")
dprint_indented = debug.dprint_factory(__name__, "indented")
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
        return parser.send(token)

    lexer = lex(lexer_action)

    try:
        for line in file:
            lexer.send(line)
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
class Expression(Token):
    def __init__(self, lineno, column, text):
        super().__init__(lineno, column, text, display="<expression>")
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
    token = yield
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
def parse_set(action, token):
    # "set" Identifier "=" Expression EndOfHeading
    entries = {}

    token = yield
    if not isinstance(token, Identifier):
        raise ExpectedGotError("name", token)
    name = token.text

    token = yield
    if not token.is_punctuation("="):
        raise ExpectedGotError("`='", token)

    token = yield "expression"
    if not isinstance(token, Expression):
        raise ExpectedGotError("expression", token)
    entries["set"] = token.text

    token = yield
    if not isinstance(token, EndOfHeading):
        raise ExpectedGotError("eol", token)

    token = yield
    if isinstance(token, IndentedText):
        raise ExpectedGotError("keyword or eof", token)

    action("global", name, entries)
    return token


@subcoroutine
def parse_macro(action, token):
    # "macro" Identifier "(" {Identifier,*} ")" ":"? EndOfHeading IndentedText?
    entries = {}

    token = yield
    if not isinstance(token, Identifier):
        raise ExpectedGotError("name of macro", token)
    name = token.text

    token = yield
    if not token.is_punctuation("("):
        raise ExpectedGotError("`('", token)

    param_names = []
    while True:
        token = yield
        if token.is_punctuation(")"):
            break
        elif isinstance(token, Identifier):
            param_names.append(token.text)
        else:
            raise ExpectedGotError("parameter name or `)'", token)

        token = yield
        if token.is_punctuation(")"):
            break
        elif token.is_punctuation(","):
            continue
        else:
            raise ExpectedGotError("`,' or `)'", token)
    entries["params"] = param_names

    expected = "`:' or eol"
    token = yield
    if token.is_punctuation(":"):
        expected = "eol"
        token = yield
    if not isinstance(token, EndOfHeading):
        raise ExpectedGotError(expected, token)

    token = yield
    text, token = yield from parse_optional_indented_text(token)
    if text:
        entries["macro"] = text

    action("global", name, entries)
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
        expected = ([e.expected] if isinstance(e.expected, str)
                    else list(e.expected))
        e.expected = expected + ["`:'", "eol"]
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

    if kind == "dimension":
        try:
            pairs, token = yield from parse_pairs(("format",), token)
            entries.update(pairs)
        except ExpectedGotError as e:
            expected = ([e.expected] if isinstance(e.expected, str)
                        else list(e.expected))
            e.expected = expected + ["`:'", "eol"]
            raise e

    expected = "`:', or eol"
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

    if kind == "dimension":
        try:
            pairs, token = yield from parse_pairs(("format",), token)
            entries.update(pairs)
        except ExpectedGotError as e:
            expected = ([e.expected] if isinstance(e.expected, str)
                        else list(e.expected))
            e.expected = expected + ["`:'", "eol"]
            raise e

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
# Lexer
#

# Usage:
# lexer = lex(action)  # action is a callable accepting Tokens
# for line in file:
#     lexer.send(line)
# lexer.close()

# In some cases, action(token) may return control keywords instructing the
# lexer to switch modes. Currently, the only cases is "expression" in response
# to a Punctuation("=").

hspace = " \t"


class RestOfLine:
    # A representation of the unconsumed portion of the current line.

    def __init__(self, line, lineno):
        self.line = line
        self.lineno = lineno
        self.column = 1

    def __len__(self):
        return len(self.line)

    def peek(self, length=1):
        return self.line[:length]

    def consume(self, length=1):
        ret = self.line[:length]
        self.line = self.line[length:]
        self.column += len(ret)
        return ret

    def match(self, pattern):
        return re.match(pattern, self.line)

    def is_empty(self):
        return not self.line.strip()

    def is_comment(self):
        return self.line.lstrip().startswith("#")

    def is_empty_or_comment(self):
        s = self.line.strip()
        if not s or s.startswith("#"):
            return True
        return False


@coroutine
def lex(action):
    # Receives lines as bare strings.
    lineno_counter = itertools.count(1)
    line_lexer = lex_lines(action)
    line = ""
    lineno = 1
    while True:
        try:
            line = (yield).rstrip("\n")
        except GeneratorExit:
            line_lexer.close()
            action(EndOfInput(lineno, len(line)))
            return
        else:
            lineno = next(lineno_counter)
            dprint_line("lexer received line", "{:d}:".format(lineno), line)
            line_lexer.send(RestOfLine(line, lineno))


@coroutine
def lex_lines(action):
    # Receives RestOfLine objects via yield.
    while True:
        rol = yield
        if rol.is_empty_or_comment():
            continue
        if rol.peek() in hspace:
            raise SyntaxError("unexpected indent", lineno=lineno)
        # Now we have the first unindented line.
        break

    while True:
        rol = yield from lex_heading(action, rol)
        rol = yield from lex_indented_lines(action, rol)


@subcoroutine
def lex_heading(action, rol):
    paren_stack = []
    while True:
        if paren_stack:
            while rol.is_empty_or_comment():
                # Let parser deal with eof error within parentheses.
                rol = yield
        elif not len(rol):
            break

        hspace_pattern = "[{}]+".format(hspace)
        m = rol.match(hspace_pattern)
        if m:
            rol.consume(m.end())
            continue

        ch = rol.peek()
        if ch in "[]()=.,:":
            ctrl = action(Punctuation(rol.lineno, rol.column, ch))
            if ch in "[(":
                paren_stack.append({"[": "]", "(": ")"}[ch])
            elif ch in ")]":
                # As a lexer we tolerate unmatched parens (which lead to parse
                # errors anyway), but keep track of the matching ones.
                if paren_stack and ch == paren_stack[-1]:
                    paren_stack.pop()
            rol.consume()

            if ch == "=" and ctrl == "expression":
                rol = yield from lex_expression(action, rol)
                break

            continue

        if re.match("[A-Za-z_]", ch):
            rol = yield from lex_word(action, rol)
            continue

        if ch in "'\"":
            rol = yield from lex_quoted_string(action, rol)
            continue

        if ch == "\\" and len(rol) == 1:
            # Backslash at end of line: continuation.
            try:
                rol = yield
            except GeneratorExit:
                raise SyntaxError("line continuation at end of input",
                                  rol.lineno, rol.column)
            continue

        # Unexpected character.
        raise SyntaxError("unexpected character ({})".format(ch),
                          rol.lineno, rol.column)

    action(EndOfHeading(rol.lineno, rol.column))

    rol = yield
    return rol


@subcoroutine
def lex_expression(action, rol):
    # A Python-style expression. Can span newlines that occur within
    # parentheses and triple-quoted strings. Newlines can also be escaped with
    # a backslash.
    start_lineno, start_column = rol.lineno, rol.column
    chunks = []
    paren_stack = []
    while True:
        if paren_stack:
            try:
                while rol.is_empty_or_comment():
                    rol = yield
                    chunks.append("\n")
                    dprint_expression("epxression chunk", chunks[-1])
            except GeneratorExit:
                break
        elif not len(rol):
            break

        m = rol.match("[^\"\'\\\\({[\\]})]+")
        if m:
            chunks.append(rol.consume(m.end()))
            dprint_expression("epxression chunk", chunks[-1])
            continue

        ch = rol.peek()
        if ch in "{[(":
            chunks.append(rol.consume())
            dprint_expression("epxression chunk", chunks[-1])
            paren_stack.append({"{": "}", "[": "]", "(": ")"}[ch])
            continue

        if ch in ")]}":
            chunks.append(rol.consume())
            dprint_expression("epxression chunk", chunks[-1])
            if paren_stack and ch == paren_stack[-1]:
                paren_stack.pop()
            continue

        if ch in "'\"":
            rol = yield from lex_quoted_string(lambda s: chunks.append(s),
                                               rol, raw=True)
            dprint_expression("epxression chunk", chunks[-1])
            continue

        if ch == "\\" and len(rol) == 1:
            try:
                rol = yield
            except GeneratorExit:
                raise SyntaxError("line continuation at end of input",
                                  rol.lineno, rol.column)
            continue

    action(Expression(start_lineno, start_column, "".join(chunks)))
    return rol


@subcoroutine
def lex_indented_lines(action, rol):
    indented_lines = [] # Includes empty lines and indented comment lines.
    start_lineno = rol.lineno
    while True:
        if len(rol) and rol.peek() not in hspace:
            # A non-indented line.
            if not rol.is_comment():
                break

        if not rol.is_comment():
            dprint_indented("indented line", rol.line)
            indented_lines.append(rol.line)

        try:
            rol = yield
        except GeneratorExit:
            break

    if indented_lines:
        text = process_indented_lines(indented_lines)
        if text:
            dprint_indented("processed indented", text)
            action(IndentedText(start_lineno, 0, text))

    return rol


_qualified_word_pattern = \
        re.compile(r"[A-Za-z_][0-9A-Za-z_]*(\.[A-Za-z_][0-9A-Za-z_]*)*")
_keywords = [
            "defs",
            "set",
            "macro",
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
@subcoroutine
def lex_word(action, rol):
    m = rol.match(_qualified_word_pattern)
    assert m
    word = rol.consume(m.end())
    if m.group(1): # Contains `.'
        action(QualifiedIdentifier(rol.lineno, rol.column, word))
    elif word in _keywords:
        action(Keyword(rol.lineno, rol.column, word))
    else:
        action(Identifier(rol.lineno, rol.column, word))
    return rol
    yield


@subcoroutine
def lex_quoted_string(action, rol, raw=False):
    start_lineno, start_column = rol.lineno, rol.column
    chunks = []

    quote_char = rol.consume()
    assert quote_char in ("'", '"')
    if raw:
        chunks.append(quote_char)

    triple_quote = False
    if rol.peek(2) == quote_char * 2:
        rol.consume(2)
        triple_quote = True
        if raw:
            chunks.append(quote_char * 2)

    # Now we are just after the opening quote.
    while True:
        while not len(rol):
            if not triple_quote:
                raise SyntaxError("unterminated string",
                                  start_lineno, start_column)
            chunks.append("\n")
            try:
                rol = yield
            except GeneratorExit:
                raise SyntaxError("unterminated string",
                                  start_lineno, start_column)

        m = rol.match("[^\"\'\\\\]+")
        if m:
            chunks.append(rol.consume(m.end()))
            continue

        ch = rol.peek()
        if ch == quote_char:
            if not triple_quote:
                rol.consume()
                if raw:
                    chunks.append(quote_char)
                break
            elif rol.peek(3) == quote_char * 3:
                rol.consume(3)
                if raw:
                    chunks.append(quote_char * 3)
                break

        if ch == "\\":
            chunks.append(get_string_escape(rol, raw=raw))
            continue

        chunks.append(rol.consume()) # A non-triple quote.

    if raw:
        dprint_string("raw string chunks", chunks)
        action("".join(chunks))
    else:
        dprint_string("string chunks", chunks)
        action(String(start_lineno, start_column, "".join(chunks)))

    return rol


_str_escapes = {
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
def get_string_escape(rol, raw=False):
    assert rol.consume == "\\"
    ch = rol.peek()
    if ch in _str_escapes:
        if raw:
            return "\\" + ch
        return _str_escapes[rol.consume()]
    if ch and ch in "01234567xuU":
        if raw:
            return "\\" + get_unicode_escape(rol, raw=True)
        return get_unicode_escape(rol)
    # Otherwise, not an escape.
    # Leave peeked ch unconsumed.
    return "\\"


def get_unicode_escape(rol, raw=False):
    # rol should be just after the backslash.
    escape_start_column = rol.column
    ch = rol.peek()

    if ch in "xuU":
        escape_ch = rol.consume()
        base = 16
        # Must have the required digits.
        n_digits = {"x": 2, "u": 4, "U": 8}[escape_ch]
        digits = rol.consume(n_digits)
        if len(digits) < n_digits:
            raise SyntaxError("invalid \\{}{} escape".
                              format(escape_ch, "X" * n_digits),
                              rol.lineno, escape_start_column)

    else:
        escape_ch = ""
        base = 8
        digits = ""
        while len(rol) and rol.peek() in "01234567" and len(digits) <= 3:
            digits += rol.consume()

    if raw:
        return escape_ch + digits

    try:
        code = int(digits, base)
    except ValueError:
        raise SyntaxError("invalid \\{}{} escape".
                          format(escape_ch, "X" * n_digits),
                          rol.lineno, escape_start_column)

    return chr(code)


def process_indented_lines(lines):
    # Remove trailing empty lines.
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return None

    # Find the common indent.
    common_space_prefix = space_prefix(lines[0])
    for line in lines[1:]:
        if not line.strip():
            continue # Ignore indent of empty lines.
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
