import os
import json
from dataclasses import dataclass

HOME = os.environ.get("HOME", "~")
USER_CFG_PATH = os.path.join(HOME, ".ai.config.json")


@dataclass
class Cfg:
  abbreviations: dict
  filename_max_words: int
  model: str
  pygments_theme: str
  saved_chats_dir: str
  system_message: str
  debug: bool = False

  @property
  def abbreviations_reverse(self):
    return {v: k for k, v in self.abbreviations.items()}


DEFAULT_CFG = {
  "abbreviations": {"user": "_U_", "assistant": "_A_", "system": "_S_"},
  "filename_max_words": 10,
  "model": "gpt-3.5-turbo",
  "pygments_theme": "monokai",
  "saved_chats_dir": "~/ai-chats",
  "system_message": "You are my kind and helpful assistant.",
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
