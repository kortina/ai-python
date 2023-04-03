import datetime
import sqlite3
from pathlib import Path

from openai import ChatCompletion

from libai import CFG

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
INSERT_ROW = "INSERT INTO chats (text, role, model, created_at, token_count, md_path) VALUES (?,?,?,?,?,?)"


def save_sqlite(
    user_prompt: str, completion: ChatCompletion, system_message: str, path, query_time: datetime, response_time: datetime
):
    db_path = Path(CFG.saved_chats_dir) / "chats.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(CREATE_TABLE)
    c = conn.cursor()
    c.execute(
        INSERT_ROW,
        (
            user_prompt.strip(),
            "user",
            completion["model"],
            query_time,
            completion["usage"]["prompt_tokens"],
            path,
        ),
    )
    for choice in completion["choices"]:
        c.execute(
            INSERT_ROW,
            (
                choice["message"]["content"].strip(),
                choice["message"]["role"],
                completion["model"],
                response_time,
                completion["usage"]["completion_tokens"],
                path,
            ),
        )
    if system_message:
        c.execute(
            INSERT_ROW,
            (
                system_message,
                "system",
                completion["model"],
                query_time,
                0,
                path,
            ),
        )
    conn.commit()
    conn.close()
