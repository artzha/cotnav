from enum import Enum
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

# Canonical content type enum
class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

# Common roles enum
class Role(str, Enum):
    USER = "user"
    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"

@dataclass
class ChatQuery:
    type: str = ContentType.TEXT.value
    role: str = Role.USER.value
    content: Any = field(default_factory=str)

    def __post_init__(self) -> None:
        # Normalize simple content values into a list for consistent downstream use.
        if isinstance(self.content, (str, dict)):
            self.content = self.content

    def as_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "role": self.role, "content": self.content}