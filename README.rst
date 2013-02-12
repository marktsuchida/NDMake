NDMake
======

**NDMake** (from "N-dimensional make") is a tool for automating tasks that
involve running commands in multiple steps, at each step iterating over
multiple parameters, taking multiple inputs, and producing multiple output
files. Like ``make``, it can check the file modification dates and run the
commands only when updates are necessary.

NDMake is in the early stage of development. There is rudimentary built-in help
for the ``ndmake`` command, but documentation for the input file (ndmakefile)
syntax does not yet exist. Functionality may change in a backward-incompatible
manner.

NDMake is written in Python and requires Python 3.3 or greater.
