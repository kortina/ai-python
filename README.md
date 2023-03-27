# a.py

python CLI for an AI Assistant

Currently built on `gpt-3.5-turbo` / [openai-python](https://github.com/openai/openai-python).

## Installation

Get your account's secret key from [openai](https://platform.openai.com/account/api-keys) and export it as an environment variable in your shell:

```zsh
export OPENAI_API_KEY='sk-...'
```

Built with [click][] with Shell Completion in mind. Add to your this `.zshrc` (or see [click][] docs for other shells):

```zsh
# assuming you have checked out at `$HOME/src/a.py/`:
if [ -d "$HOME/src/a.py" ] ; then
  # add this directory to your path
  export PATH="$PATH:$HOME/src/a.py"
  # if you use a.py instead of an alias, change `a` to `a.py`
  eval "$(_A_COMPLETE=zsh_source a)"
fi
```

## Usage

## Contribution

[click]: https://click.palletsprojects.com/en/8.1.x/shell-completion/
