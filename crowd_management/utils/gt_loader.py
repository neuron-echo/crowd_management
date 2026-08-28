"""Ground truth loader for HAJJv2 dataset annotation CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class HajjGroundTruth:
    """Indexed ground-truth bounding boxes and pedestrian counts per frame."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.frame_boxes: dict[int, list[dict[str, Any]]] = {}
        self.frame_counts: dict[int, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_no = int(row.get("Frame_No", 0))
                    x = float(row.get("X", 0))
                    y = float(row.get("Y", 0))
                    w = float(row.get("Width", 0))
                    h = float(row.get("Height", 0))
                    cls_name = row.get("Classes", "").strip()
                    obj_name = row.get("Obj_Name", "").strip()

                    box = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "class": cls_name,
                        "name": obj_name,
                    }
                    if frame_no not in self.frame_boxes:
                        self.frame_boxes[frame_no] = []
                    self.frame_boxes[frame_no].append(box)
                except (ValueError, TypeError):
                    continue

        for frame_no, boxes in self.frame_boxes.items():
            self.frame_counts[frame_no] = len(boxes)

    def get_count_for_frame(self, frame_no: int) -> int | None:
        """Return the ground truth pedestrian count for a given frame index (1-based), or None if unannotated."""
        return self.frame_counts.get(frame_no)

    def get_boxes_for_frame(self, frame_no: int) -> list[dict[str, Any]]:
        """Return bounding box list for a given frame index."""
        return self.frame_boxes.get(frame_no, [])

    @property
    def total_annotated_frames(self) -> int:
        return len(self.frame_counts)
