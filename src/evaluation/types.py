from dataclasses import dataclass


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]
