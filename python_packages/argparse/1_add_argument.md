# Usage of `add_argument`

Arguments to `ArgumentParser.add_argument`:

#### name or flags

The first arguments passed to `add_argument` must be a series of flags, or a simple argument name. A __positional argument__ comes before any __optional argument__ and does not need a flag to indicate its presence.

```python
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-f', '--foo')
>>> parser.add_argument('bar')
>>> parser.parse_args(['BAR'])
Namespace(bar='BAR', foo=None)
>>> parser.parse_args(['BAR', '--foo', 'FOO'])
Namespace(bar='BAR', foo='FOO')
>>> parser.parse_args(['--foo', 'FOO'])
usage: PROG [-h] [-f FOO] bar
PROG: error: the following arguments are required: bar
```

#### action

`'store'`: stores the value that comes after the flag (default).

`'store_const'`: stores a predefined constant value (keyword `const`).

`'store_true'`: stores `True`.

`'store_false'`: stores `False`.

`'append'`: append the values to the list (same flag can be used repeatedly).

`'append_const'`: append a predefined constant (keyword `const`) value to a specified list (keyword `dest`)

#### nargs

`None`: stores the value directly.

`N` (where `N` is an integer): N arguments will be gathered into a list.

`'?'`:  zero or one argument.

`'*'`:  zero or multiple arguments.

`'+'`: multiple but at least one argument.

`argparse.REMAINDER`: all the remaining arguments.

Values other than `None` for `nargs` will gather values into a list, even if it's `nargs=1`.

#### metavar

The name for the object displayed in the help message.

#### dest

The name for the object actually in the namespace.