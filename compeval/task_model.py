"""
COMPEVAL Task Model - Data structures for tasks, task sets, and evaluation dimensions.
"""

from typing import Any, List, Dict, Optional, NamedTuple, Iterator, Tuple
from enum import Enum
import json


class TaskFamily(str, Enum):
    """The four core types of compositional tasks."""

    INTRA_STRUCTURE = "intra-structure composition"
    INTER_STRUCTURE = "inter-structure composition"
    FIELD_ARITHMETIC = "field arithmetic"
    RULE_INDUCTION = "rule induction"


class CompositionDimension(str, Enum):
    """Standard dimensions of compositional generalization (from Hupkes et al., 2020)."""

    GENERAL = "general"
    SYSTEMATICITY = "systematicity"
    PRODUCTIVITY = "productivity"
    SUBSTITUTIVITY = "substitutivity"
    OVERGENERALIZATION = "overgeneralization"


class Task(NamedTuple):
    """A single evaluation instance."""

    task_id: str
    prompt: str
    answer: str
    answer_raw: Any
    depth: int
    family: TaskFamily
    dimension: CompositionDimension = CompositionDimension.GENERAL
    structures: List[str] = []
    metadata: Dict[str, Any] = {}
    solution_trace: Optional[List[Tuple[str, Any]]] = None


class TaskSet:
    """A collection of tasks for a benchmark run."""

    def __init__(
        self,
        tasks: List[Task],
        name: str = "compeval",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tasks = tasks
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        self._task_map = {task.task_id: task for task in tasks}

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, key) -> Task:
        if isinstance(key, int):
            return self.tasks[key]
        return self._task_map[key]

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def to_jsonl(self, path: str) -> None:
        """Save the task set to a JSONL file."""
        with open(path, "w") as f:
            for task in self.tasks:
                d = task._asdict()
                # Convert enums to strings for JSON serialization
                d["family"] = d["family"].value if hasattr(d["family"], "value") else d["family"]
                d["dimension"] = d["dimension"].value if hasattr(d["dimension"], "value") else d["dimension"]
                f.write(json.dumps(d) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "TaskSet":
        """Load a task set from a JSONL file."""
        tasks = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                data["family"] = TaskFamily(data["family"])
                data["dimension"] = CompositionDimension(data["dimension"])
                tasks.append(Task(**data))
        return cls(tasks)

    def summary(self) -> str:
        """Return a string summarizing the task set composition."""
        counts: Dict[Tuple[str, int], int] = {}
        for task in self.tasks:
            key = (task.family.value, task.depth)
            counts[key] = counts.get(key, 0) + 1

        summary_str = f"TaskSet '{self.name}' Summary ({len(self.tasks)} tasks total):\n"
        for (family, depth), count in sorted(counts.items()):
            summary_str += f"  - {family} (Depth {depth}): {count} tasks\n"
        return summary_str
