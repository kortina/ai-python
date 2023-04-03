#!/usr/bin/env python3
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import MarkdownLexer
import click
import datetime
import openai
import os
import re

from libai.config import CFG
from libai.sqlite_util import save_sqlite

def _chat_path(chat):
    return os.path.join(CFG.saved_chats_dir, chat)


def _filter_chats(chats):
    return list(set(chats).difference({".DS_Store", "chats.db"}))


def _chats():
    if not os.path.exists(CFG.saved_chats_dir):
        return []
    chats = os.listdir(CFG.saved_chats_dir)
    chats = _filter_chats(chats)
    chats.sort()
    return chats


def _chats_by_modified_date_desc():
    if not os.path.exists(CFG.saved_chats_dir):
        return []
    chats = os.listdir(CFG.saved_chats_dir)
    chats = _filter_chats(chats)
    chats.sort(key=lambda x: os.path.getmtime(os.path.join(CFG.saved_chats_dir, x)))
    chats.reverse()
    return chats


def _most_recent_chat():
    return _chats_by_modified_date_desc()[0]


CHATS_CHOICES = click.Choice(_chats())


@click.command(help="cli for ai assistant")
@click.argument("prompt", type=str, required=False)
@click.option("--chat", help="chat file name to load as context", type=CHATS_CHOICES)
@click.option(
    "-l",
    "--ls",
    help="list chats",
    is_flag=True,
)
@click.option(
    "-r",
    "--ls-recent",
    help="list chats by recency",
    is_flag=True,
)
@click.option(
    "-vv",
    "--verbose",
    help="debug verbose output",
    is_flag=True,
)
@click.option(
    "--cat",
    help="cat a chat",
    is_flag=True,
)
@click.option(
    "-c",
    "--rc",
    help="use most recent chat as context",
    is_flag=True,
)
@click.option(
    "-i",
    "--interactive",
    help="interactive mode",
    is_flag=True,
)
def main(prompt, chat, ls, ls_recent, verbose, cat, rc, interactive):
    """Ask gpt about this prompt."""
    if rc and chat:
        print("[ERROR]: Cannot specify both --rc and --chat.")
        exit(1)
    elif rc:
        _dbg(f"Using most recent chat: {chat}", "LOAD")
        chat = _most_recent_chat()

    if verbose:
        CFG.debug = True
    if cat:
        if not chat:
            print("cat requires a chat name.")
            return
        markdown = _load_chat(chat)
        highlighted_text = _highlight(markdown)
        print(highlighted_text)

    elif ls:
        print("\n".join(_chats()))
    elif ls_recent:
        print("\n".join(_chats_by_modified_date_desc()))
    elif prompt:
        ask(prompt, chat, interactive)
    else:
        print("Provide prompt or specify another option.")


def _highlight(text: str) -> str:
    # This will output the Markdown text with color coding for headings,
    # bold, and italic text. You can customize the colors and formatting by
    # modifying the `TerminalFormatter` options.
    # custom_style = TerminalFormatter().style.copy()
    # custom_style.update({"underline": "ansigreen"})
    return highlight(
        text, MarkdownLexer(), Terminal256Formatter(bg="dark", style=CFG.pygments_theme)
    )


def _dbg(msg: str, label=""):
    if CFG.debug:
        label = f"{label}".ljust(6)
        print(f"[DEBUG: {label} ]:  {msg}")


def _slug(prompt: str, with_date=True) -> str:
    slug_name = prompt
    # replace all non-alpha-numeric with -
    slug_name = re.sub(r"[^a-zA-Z0-9]+", "-", slug_name)

    # remove leading and trailing -
    slug_name = slug_name.strip("-")
    return slug_name


def _filename(prompt: str) -> str:
    # truncate filename to 10 words:
    slug_name = " ".join(prompt.split(" ")[: CFG.filename_max_words])
    # slugify and lowercase:
    slug_name = _slug(slug_name).lower()
    # add date:
    dt = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
    return f"{slug_name}--{dt}.md"


def _parse_markdown(markdown: str) -> dict:
    title = None
    messages = []
    speaker = None
    message = ""
    for line in markdown.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:]
            continue
        elif (
            line.startswith("_U_:")
            or line.startswith("_A_:")
            or line.startswith("_S_:")
        ):
            if speaker and message != "":
                messages.append({"role": speaker, "content": message.strip()})
                message = ""
            speaker = CFG.abbreviations_reverse[line[:3]]
            continue
        else:
            message += f"{line}\n"
    if speaker and message != "":
        messages.append({"role": speaker, "content": message.strip()})
    return {
        "title": title,
        "messages": messages,
    }


def _to_markdown(title: str, messages: list) -> str:
    """
    Given a title and a list of messages, return a markdown string.
    """
    markdown = f"# {title}\n"
    for message in messages:
        speaker = CFG.abbreviations[message["role"]]
        content = message["content"]
        markdown += f"\n{speaker}:\n{content.strip()}\n"
    return markdown.strip()


def _filepath(title) -> str:
    filename = _filename(title)
    return os.path.join(CFG.saved_chats_dir, filename)


def _save_markdown(messages: list, filepath=None):
    title = _title(messages)
    if not filepath:
        filepath = _filepath(title)
        _dbg(f"{filepath}", "CREATE")
    markdown = _to_markdown(title, messages)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)
    return filepath


# only use the first line of the prompt as the title:
def _title_from_prompt(prompt: str) -> str:
    return f"{prompt}".strip().split("\n")[0]


# use the first user message as the title:
def _title(messages: list) -> str:
    for message in messages:
        if message["role"] == "user":
            return _title_from_prompt(message["content"])
    return "NO-USER-MESSAGE"


def _load_chat(chat: str) -> str:
    path = _chat_path(chat)
    _dbg(f"{path}", "LOAD")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _gen_completion(prompt: str, chat=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_ORGANIZATION"):
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
    new_message = {"role": "user", "content": prompt.strip()}

    path = None

    _dbg(new_message["content"], "PROMPT")

    if chat:
        path = _chat_path(chat)
        markdown = _load_chat(chat)
        parsed = _parse_markdown(markdown)
        messages = parsed["messages"]
    else:
        messages = [{"role": "system", "content": CFG.system_message}]

    messages.append(new_message)

    chunk_texts = []
    for chunk in openai.ChatCompletion.create(
        model=CFG.model,
        messages=messages,
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            chunk_texts.append(content)
            print(content, end='')

    response = ''.join(chunk_texts)
    messages.append({"role": "assistant", "content": response})
    save_path = _save_markdown(messages, filepath=path)
    return response, messages, save_path


import sys
def ask(prompt: str, chat=None, interactive=False):
    query_time = datetime.datetime.now()
    response, messages, path = _gen_completion(prompt, chat)
    response_time = datetime.datetime.now()

    count = response.count("\n")
    for _ in range(count):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

    _dbg("----------------------------------------\n")
    print("\n")
    highlighted_text = _highlight(response)
    print(highlighted_text, end='')
    print("\n")
    _dbg("----------------------------------------")

    # save_sqlite(
    #     prompt,
    #     completion,
    #     system_message=CFG.system_message if not chat else None,
    #     path=path,
    #     query_time=query_time,
    #     response_time=response_time,
    # )


if __name__ == "__main__":
    main()
