# ai-python

python CLI for an AI Assistant

Currently built on `gpt-3.5-turbo` / [openai-python](https://github.com/openai/openai-python).

## Installation

Get your account's secret key from [openai](https://platform.openai.com/account/api-keys) and export it as an environment variable in your shell:

```zsh
export OPENAI_API_KEY='sk-...'
```

Built with [click][] with Shell Completion in mind. Add to your this `.zshrc` (or see [click][] docs for other shells):

```zsh
if [ -d "$HOME/src/ai-python" ] ; then
  export PATH="$PATH:$HOME/src/ai-python"
  eval "$(_AI_COMPLETE=zsh_source ai)"
  # I also alias to just `a` ;)
  alias a="ai"
fi
```

## Usage

## Contribution

[click]: https://click.palletsprojects.com/en/8.1.x/shell-completion/
