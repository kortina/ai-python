#!/usr/bin/env python3
"""
pip uninstall open-clip-torch tensorboardx
pip install - q -U google-generativeai
"""
import datetime
import json
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
import google.generativeai as genai
import openai
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import MarkdownLexer

HOME = os.environ.get("HOME", "~")
USER_CFG_PATH = os.path.join(HOME, ".ai.config.json")


class Api(Enum):
    GOOGLE = "google"
    OPENAI = "openai"


@dataclass
class Cfg:
    filename_max_words: int
    google_model: str
    openai_model: str
    pygments_theme: str
    saved_chats_dir: str
    system_message: str
    debug: bool = False
    api: Api = Api.GOOGLE

    @property
    def abbreviations_reverse(self):
        return {v: k for k, v in self.abbreviations.items()}


DEFAULT_CFG = {
    "filename_max_words": 10,
    "openai_model": "gpt-3.5-turbo",
    "google_model": "gemini-1.5-pro",
    "pygments_theme": "monokai",
    "saved_chats_dir": "~/ai-chats",
    "system_message": "You are my kind and helpful assistant.",
}


class Speaker:
    USER: str = "USER"
    ASSISTANT: str = "ASSISTANT"
    SYSTEM: str = "SYSTEM"


ABBREVIATIONS = {
    "_U_": Speaker.USER,
    "_USER_": Speaker.USER,
    "_A_": Speaker.ASSISTANT,
    "_ASSISTANT_": Speaker.ASSISTANT,
    "_S_": Speaker.SYSTEM,
    "_SYSTEM_": Speaker.SYSTEM,
    "_I_": Speaker.SYSTEM,
    "_INSTRUCTIONS_": Speaker.SYSTEM,
}


def _load_cfg():
    cfg = DEFAULT_CFG
    if os.path.exists(USER_CFG_PATH):
        with open(USER_CFG_PATH) as f:
            user_cfg = json.load(f)
            for k, v in cfg.items():
                if k in user_cfg:
                    cfg[k] = user_cfg[k]

    cfg["saved_chats_dir"] = os.path.expanduser(cfg["saved_chats_dir"])
    # mkdir -p saved_chats_dir:
    os.makedirs(cfg["saved_chats_dir"], exist_ok=True)

    return Cfg(**cfg)


CFG = _load_cfg()


def _chat_path(chat):
    return os.path.join(CFG.saved_chats_dir, chat)


def _filter_chats(chats):
    return list(set(chats).difference({".DS_Store", "chats.db", ".vscode"}))


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
API_CHOICES = click.Choice([member.value for member in Api])


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
    "--api",
    help="Choose google or openai.",
    default=Api.GOOGLE.value,
    required=False,
    type=API_CHOICES,
)
def main(prompt, chat, ls, ls_recent, verbose, cat, rc, api):
    """Ask bot about this prompt."""
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
        ask(api, prompt, chat)
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

# NB: this method MUTATES the messages list
def _append_message_to(messages, speaker, message) -> str:
    if speaker and message != "":
        messages.append(Message(role=role, content=content)
        return ""
    return message

def _parse_markdown(markdown: str) -> dict:
    title = None
    messages = []
    speaker = None
    message = ""
    for line in markdown.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:]
            continue
        elif _is_speaker_line(line):
            message = _append_message_to(messages, speaker, message.strip())
            # strip the trailing ":" from the line and get the speaker string
            speaker = ABBREVIATIONS[line[:-1]]
            continue
        else:
            message += f"{line}\n"
    message = _append_message_to(messages, speaker, message.strip())
    return {
        "title": title,
        "messages": messages,
    }


def _is_speaker(speaker: str, s: str) -> bool:
    for k, v in ABBREVIATIONS.items():
        if v == speaker:
            if s.startswith(f"{k}:"):
                return True
    return False


def _is_user(s: str) -> bool:
    return _is_speaker(Speaker.USER, s)


def _is_assistant(s: str) -> bool:
    return _is_speaker(Speaker.ASSISTANT, s)


def _is_system(s: str) -> bool:
    return _is_speaker(Speaker.SYSTEM, s)


def _is_speaker_line(s: str) -> bool:
    return _is_user(s) or _is_assistant(s) or _is_system(s)


def _to_markdown(title: str | None, messages: list) -> str:
    """
    Given a title and a list of messages, return a markdown string.
    """
    markdown = ""
    if title:
        markdown += f"# {title}\n"
    for message in messages:
        speaker = CFG.abbreviations[message.role]
        content = message.content
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
        if message.role == "user":
            return _title_from_prompt(message.content)
    return "NO-USER-MESSAGE"


def _load_chat(chat: str) -> str:
    path = _chat_path(chat)
    _dbg(f"{path}", "LOAD")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chats
(
    id          INTEGER PRIMARY KEY,
    text        TEXT     NOT NULL,
    role        TEXT     NOT NULL,
    model       TEXT     NOT NULL,
    created_at  DATETIME NOT NULL,
    token_count INTEGER  NOT NULL,
    md_path     TEXT DEFAULT NULL
);
"""
INSERT_ROW = (
    "INSERT INTO chats (text, role, model, created_at, token_count, md_path) VALUES (?,?,?,?,?,?)"
)


def save_sqlite(message, response, system_message, path):
    db_path = Path(CFG.saved_chats_dir) / "chats.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(CREATE_TABLE)
    c = conn.cursor()
    c.execute(
        INSERT_ROW,
        (
            message.content,
            message.role,
            response.model,
            response.query_time,
            response.token_count,
            path,
        ),
    )
    if response.choices:
        for choice in response.choices:
            message = choice.message
            c.execute(
                INSERT_ROW,
                (
                    message.content.strip(),
                    message.role,
                    response.model,
                    response.response_time,
                    response.token_count,
                    path,
                ),
            )
    if system_message:
        c.execute(
            INSERT_ROW,
            (
                system_message.content,
                system_message.role,
                response.model,
                response.query_time,
                0,
                path,
            ),
        )
    conn.commit()
    conn.close()


@dataclass
class Message:
    role: str
    content: Optional[str] = None


@dataclass
class ResponseChoice:
    message: Message


@dataclass
class Response:
    model: str
    choices: List[ResponseChoice]
    token_count: int
    query_time: datetime.datetime
    response_time: datetime.datetime

    def text(self):
        return self.choices[0].message.get("content")


class ApiClient(ABC):
    prompt: Optional[str] = None
    chat: Optional[str] = None

    @abstractmethod
    def __init__(self, chat: Optional[str] = None) -> None:
        self.chat = chat

    @abstractmethod
    def response(self, messages):
        raise NotImplementedError

    @abstractmethod
    def model(self):
        raise NotImplementedError

    def ask(self, _prompt: str):
        self.prompt = _prompt.strip()
        new_message = Message(role="user", content=self.prompt)
        _dbg(self.prompt, "PROMPT")

        messages = self.messages()

        # append our question:
        messages.append(new_message)

        response = self.response(messages)

        text = response.choices[0].message.content
        _dbg("----------------------------------------\n")
        print("\n")
        highlighted_text = _highlight(text)
        print(highlighted_text)
        print("\n")
        _dbg("----------------------------------------")

        # append the response:
        messages.append(Message(role="assistant", content=text))

        print("_save_markdown........")
        print("save_sqlite...........")
        return

        _save_markdown(messages, filepath=self.chat_path())

        save_sqlite(
            new_message,
            response,
            system_message=self.system_message(),
            path=self.chat_path(),
        )

    def system_message(self):
        if not self.chat:
            return Message(role="system", content=CFG.system_message)

    def messages(self) -> List[Message]:
        if self.chat:
            return self.parsed_markdown()["messages"]
        else:
            return [self.system_message()]

    def chat_path(self):
        if self.chat:
            return _chat_path(self.chat)

    def markdown(self):
        if self.chat:
            return _load_chat(self.chat)

    def parsed_markdown(self):
        if self.chat:
            return _parse_markdown(self.markdown())


class OpenAI(ApiClient):
    def __init__(self, chat: str | None = None) -> None:
        super().__init__(chat)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_ORGANIZATION"):
            openai.organization = os.getenv("OPENAI_ORGANIZATION")

    def model(self):
        return CFG.openai_model

    def response(self, messages):
        query_time = datetime.datetime.now()
        completion = openai.ChatCompletion.create(
            model=self.model(),
            messages=messages,
        )
        response_time = datetime.datetime.now()
        choices = []
        for choice in completion.choices:
            m = Message(
                role=choice["message"]["role"],
                content=choice["message"]["content"],
            )
            rc = ResponseChoice(message=m)
            choices.append(rc)

        return Response(
            model=completion["model"],
            choices=choices,
            token_count=completion["usage"]["completion_tokens"],
            query_time=query_time,
            response_time=response_time,
        )


class Google(ApiClient):
    def __init__(self, chat: str | None = None) -> None:
        super().__init__(chat)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)

    def _list_models(self):
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)

    def model(self):
        return CFG.google_model

    def response(self, messages):
        query_time = datetime.datetime.now()

        print("messages: ................")
        print(messages)
        print(_to_markdown(None, messages))
        return None
        model = genai.GenerativeModel

        completion = openai.ChatCompletion.create(
            model=self.model(),
            messages=messages,
        )
        response_time = datetime.datetime.now()
        choices = []
        for choice in completion.choices:
            m = Message(
                role=choice["message"]["role"],
                content=choice["message"]["content"],
            )
            rc = ResponseChoice(message=m)
            choices.append(rc)

        return Response(
            model=completion["model"],
            choices=choices,
            token_count=completion["usage"]["completion_tokens"],
            query_time=query_time,
            response_time=response_time,
        )


def ask(api: Api, prompt: str, chat=None):
    prompt = prompt.strip()

    if api == Api.OPENAI.value:
        OpenAI().ask(prompt)
    elif api == Api.GOOGLE.value:
        Google().ask(prompt)
    else:
        raise ValueError(f"Api `{api}` not supported.")


if __name__ == "__main__":
    main()
