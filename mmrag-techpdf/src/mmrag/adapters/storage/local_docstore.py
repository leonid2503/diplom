import pickle
from pathlib import Path
from typing import Dict, List, Optional

from ...domain.artifact import Artifact


class LocalDocStore:
    """
    Simple in-memory artifact store persisted to disk via pickle.

    The FAISS index stores only integer row IDs; this store maps those
    logical artifact IDs (strings) to the full Artifact objects.
    """

    def __init__(self):
        self._store: Dict[str, Artifact] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(self, artifact: Artifact) -> None:
        self._store[artifact.id] = artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._store.get(artifact_id)

    def get_many(self, artifact_ids: List[str]) -> List[Artifact]:
        return [self._store[aid] for aid in artifact_ids if aid in self._store]

    def get_all(self) -> List[Artifact]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._store, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._store = pickle.load(f)
