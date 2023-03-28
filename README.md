# ai-python

A python CLI for an AI Assistant.

Currently built on `gpt-3.5-turbo` / [openai-python](https://github.com/openai/openai-python).

## Installation

Get your account's secret key from [openai](https://platform.openai.com/account/api-keys) and export it as an environment variable in your shell:

```zsh
# You should probably put this in your .zshrc or .bashrc:
export OPENAI_API_KEY='sk-...'
```

Clone the repo (`pip install` coming soon!):

```zsh
cd ~/src
git clone https://github.com/kortina/ai-python
```

Built with [click][] with Shell Completion in mind. Add to your this `.zshrc` (or see [click][] docs for other shells):

```zsh
if [ -d "$HOME/src/ai-python" ] ; then
  export PATH="$PATH:$HOME/src/ai-python"
  eval "$(_AI_COMPLETE=zsh_source ai)"
  # I also alias to just `a`  and `c`;)
  alias a="ai"
  alias c="ai --rc"
fi
```

Reload your shell.

## Usage

```
Usage: ai [OPTIONS] [PROMPT]

  cli for ai assistant

Options:
  --chat [name-of-chat-file.md]   chat file name to load as context
  -l, --ls                        list chats
  -r, --ls-recent                 list chats by recency
  -vv, --verbose                  debug verbose output
  --cat                           cat a chat
  -c, --rc                        use most recent chat as context
  --help                          Show this message and exit.
```

## Configuration

Save your default configuration to `~/.ai.config.json`, eg:

```json
{
  "abbreviations": { "user": "_U_", "assistant": "_A_", "system": "_S_" },
  "filename_max_words": 10,
  "saved_chats_dir": "~/ai-chats",
  "model": "gpt-3.5-turbo",
  "system_message": "You are my kind and helpful assistant."
}
```

## Screenshots

<img width="850" alt="example output" src="https://user-images.githubusercontent.com/5924/228105609-baabbc8e-cdee-4367-82ae-6c2f7b194268.png">

[click]: https://click.palletsprojects.com/en/8.1.x/shell-completion/
