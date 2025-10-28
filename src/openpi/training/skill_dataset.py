"""Dataset for skill-segmented demonstrations stored as videos and parquet tables."""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
from pathlib import Path
from typing import Mapping
from typing import Sequence
from typing import TypedDict

import imageio.v2 as imageio
import numpy as np
import polars as pl


logger = logging.getLogger(__name__)


class _SkillAnnotation(TypedDict):
    skill_idx: int
    frame_duration: Sequence[int]
    skill_description: Sequence[str]
    object_id: Sequence[Sequence[str]]
    manipulating_object_id: Sequence[str]
    memory_prefix: Sequence[str]
    spatial_prefix: Sequence[str]
    skill_type: Sequence[str]


@dataclasses.dataclass
class _SkillEntry:
    episode_index: int
    start: int
    end: int
    prompt: str
    skill_idx: int


@dataclasses.dataclass
class _EpisodeData:
    actions: np.ndarray
    states: np.ndarray


class _VideoReader:
    """Video reader backed by ``imageio`` with random access support."""

    def __init__(self, path: Path):
        self._path = path
        try:
            self._reader = imageio.get_reader(str(path))
        except FileNotFoundError as exc:  # pragma: no cover - bubble up clearer error
            raise FileNotFoundError(f"Failed to open video file: {path}") from exc
        self._frame_count = self._infer_frame_count()
        if self._frame_count <= 0:
            raise ValueError(f"Video file {path} does not report a positive frame count.")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self, frame_index: int) -> np.ndarray:
        frame_index = int(np.clip(frame_index, 0, self._frame_count - 1))
        try:
            frame = self._reader.get_data(frame_index)
        except IndexError as exc:  # pragma: no cover - should be clipped
            raise RuntimeError(f"Failed to decode frame {frame_index} from {self._path}.") from exc
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame

    def close(self) -> None:
        if hasattr(self, "_reader") and self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self) -> None:  # noqa: D401 - cleanup
        self.close()

    def _infer_frame_count(self) -> int:
        try:
            count = int(self._reader.count_frames())
            if count > 0 and count < 10**9:  # Avoid infinite counts reported by some codecs.
                return count
        except (RuntimeError, AttributeError, TypeError):
            pass
        meta = self._reader.get_meta_data()
        fps = float(meta.get("fps", 0))
        duration = float(meta.get("duration", 0))
        if fps > 0 and duration > 0:
            count = int(round(fps * duration))
            if count > 0:
                return count
        nframes = meta.get("nframes")
        if isinstance(nframes, int) and nframes > 0:
            return int(nframes)
        # As a last resort we iterate once to find the length.
        count = 0
        for _ in self._reader:
            count += 1
        return count


class SkillSequenceDataset:
    """Dataset that exposes (image, state, action sequence, prompt) tuples per skill."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        dataset_info_path: str | Path,
        split: str,
        action_horizon: int,
        image_feature_map: Mapping[str, str],
        state_key: str,
        action_key: str,
        prompt_template: str | None,
        frame_selection: str,
        max_cached_episodes: int,
        max_cached_videos: int,
    ) -> None:
        self._root_dir = Path(root_dir)
        info_path = Path(dataset_info_path)
        if not info_path.is_absolute():
            info_path = self._root_dir / info_path
        with info_path.open("r") as f:
            self._dataset_info = json.load(f)
        self._action_horizon = action_horizon
        if self._action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._image_feature_map = dict(image_feature_map)
        if not self._image_feature_map:
            raise ValueError("image_feature_map must not be empty")
        self._state_key = state_key
        self._action_key = action_key
        self._prompt_template = prompt_template
        if frame_selection not in {"start", "middle", "end"}:
            raise ValueError("frame_selection must be 'start', 'middle', or 'end'")
        self._frame_selection = frame_selection
        self._chunks_size = int(self._dataset_info["chunks_size"])
        self._data_template = self._dataset_info["data_path"]
        self._video_template = self._dataset_info["video_path"]
        self._annotation_template = self._dataset_info["annotation_path"]
        self._max_cached_episodes = max_cached_episodes
        self._episode_cache: collections.OrderedDict[int, _EpisodeData] = collections.OrderedDict()
        self._max_cached_videos = max_cached_videos
        self._video_cache: collections.OrderedDict[Path, _VideoReader] = collections.OrderedDict()

        self._skill_entries = self._build_index(split)
        logger.info("SkillSequenceDataset initialized with %d samples", len(self._skill_entries))

    def __len__(self) -> int:
        return len(self._skill_entries)

    def __getitem__(self, index: int) -> dict:
        entry = self._skill_entries[index]
        episode_data = self._get_episode_data(entry.episode_index)
        action_chunk = self._slice_actions(episode_data.actions, entry.start, entry.end)
        state = episode_data.states[self._clip_frame(entry.start, episode_data.states.shape[0])]
        images = self._load_images(entry.episode_index, entry.start, entry.end)

        image_mask = {camera_key: np.array(True, dtype=np.bool_) for camera_key in images}

        return {
            "image": images,
            "image_mask": image_mask,
            "state": state.astype(np.float32),
            "prompt": entry.prompt,
            "actions": action_chunk.astype(np.float32),
        }

    def _build_index(self, split: str) -> list[_SkillEntry]:
        try:
            split_range = self._dataset_info["splits"][split]
        except KeyError as exc:
            raise KeyError(f"Split '{split}' not found in dataset info") from exc
        if isinstance(split_range, str):
            start_str, end_str = split_range.split(":")
            start_episode = int(start_str)
            end_episode = int(end_str)
        else:
            start_episode, end_episode = split_range
        skill_entries: list[_SkillEntry] = []
        for episode_index in range(start_episode, end_episode):
            annotation_path = self._resolve_annotation_path(episode_index)
            with annotation_path.open("r") as f:
                annotation = json.load(f)
            skills: Sequence[_SkillAnnotation] = annotation.get("skill_annotation", [])  # type: ignore[assignment]
            for skill in skills:
                start, end = skill["frame_duration"]
                if end <= start:
                    continue
                prompt = self._format_prompt(skill)
                skill_entries.append(
                    _SkillEntry(
                        episode_index=episode_index,
                        start=int(start),
                        end=int(end),
                        prompt=prompt,
                        skill_idx=int(skill.get("skill_idx", 0)),
                    )
                )
        if not skill_entries:
            raise ValueError("No skill annotations found for the requested split.")
        return skill_entries

    def _format_prompt(self, skill: _SkillAnnotation) -> str:
        description = " ".join(skill.get("skill_description", ())).strip()
        objects = [
            " and ".join(obj)
            for obj in skill.get("object_id", ())
            if isinstance(obj, Sequence) and obj
        ]
        manipulating = [obj for obj in skill.get("manipulating_object_id", ()) if obj]
        prefixes = [" ".join(skill.get("memory_prefix", ())), " ".join(skill.get("spatial_prefix", ()))]
        skill_type = " ".join(skill.get("skill_type", ())).strip()

        objects_text = ", ".join(filter(None, (*objects, *manipulating)))
        prefix_text = " ".join(filter(None, prefixes)).strip()
        base_prompt_parts = [description]
        if objects_text:
            base_prompt_parts.append(objects_text)
        if prefix_text:
            base_prompt_parts.append(prefix_text)
        if skill_type:
            base_prompt_parts.append(skill_type)
        default_prompt = " ".join(part for part in base_prompt_parts if part).strip()

        if self._prompt_template:
            try:
                prompt = self._prompt_template.format(
                    description=description,
                    objects=objects_text,
                    manipulating=" ".join(manipulating),
                    prefixes=prefix_text,
                    skill_type=skill_type,
                )
            except KeyError as exc:
                raise KeyError(
                    "prompt_template references unknown placeholder. Available keys: "
                    "description, objects, manipulating, prefixes, skill_type"
                ) from exc
            prompt = prompt.strip()
            if prompt:
                return prompt
        return default_prompt or description or "perform the skill"

    def _slice_actions(self, actions: np.ndarray, start: int, end: int) -> np.ndarray:
        indices = np.arange(self._action_horizon, dtype=np.int32) + start
        indices = np.clip(indices, start, end - 1)
        indices = self._clip_frame(indices, actions.shape[0])
        return actions[indices]

    def _clip_frame(self, frame: int | np.ndarray, length: int) -> int | np.ndarray:
        return np.clip(frame, 0, length - 1)

    def _load_images(self, episode_index: int, start: int, end: int) -> dict[str, np.ndarray]:
        frame_index = self._select_frame_index(start, end)
        images: dict[str, np.ndarray] = {}
        for feature_key, camera_key in self._image_feature_map.items():
            video_path = self._resolve_video_path(feature_key, episode_index)
            reader = self._get_video_reader(video_path)
            frame = reader.read(self._clip_frame(frame_index, reader.frame_count))
            images[camera_key] = frame
        return images

    def _select_frame_index(self, start: int, end: int) -> int:
        if self._frame_selection == "start":
            return start
        if self._frame_selection == "end":
            return end - 1
        # Middle frame by default.
        return start + max((end - start) // 2, 0)

    def _get_episode_data(self, episode_index: int) -> _EpisodeData:
        if episode_index in self._episode_cache:
            episode_data = self._episode_cache.pop(episode_index)
            self._episode_cache[episode_index] = episode_data
            return episode_data
        episode_data = self._load_episode_from_disk(episode_index)
        self._episode_cache[episode_index] = episode_data
        if len(self._episode_cache) > self._max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return episode_data

    def _load_episode_from_disk(self, episode_index: int) -> _EpisodeData:
        data_path = self._resolve_data_path(episode_index)
        if not data_path.exists():
            raise FileNotFoundError(f"Parquet data file not found: {data_path}")
        table = pl.read_parquet(data_path, columns=[self._action_key, self._state_key])
        if self._action_key not in table.columns:
            raise KeyError(f"Column '{self._action_key}' not found in {data_path}")
        if self._state_key not in table.columns:
            raise KeyError(f"Column '{self._state_key}' not found in {data_path}")
        actions = np.asarray(table[self._action_key].to_list(), dtype=np.float32)
        states = np.asarray(table[self._state_key].to_list(), dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected actions to have shape [T, D], got {actions.shape}")
        if states.ndim != 2:
            raise ValueError(f"Expected states to have shape [T, S], got {states.shape}")
        return _EpisodeData(actions=actions, states=states)

    def _resolve_data_path(self, episode_index: int) -> Path:
        chunk = episode_index // self._chunks_size
        relative = self._data_template.format(episode_chunk=chunk, episode_index=episode_index)
        return (self._root_dir / relative).resolve()

    def _resolve_annotation_path(self, episode_index: int) -> Path:
        chunk = episode_index // self._chunks_size
        relative = self._annotation_template.format(episode_chunk=chunk, episode_index=episode_index)
        path = (self._root_dir / relative).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return path

    def _resolve_video_path(self, feature_key: str, episode_index: int) -> Path:
        chunk = episode_index // self._chunks_size
        relative = self._video_template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=feature_key,
        )
        path = (self._root_dir / relative).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def _get_video_reader(self, path: Path) -> _VideoReader:
        if path in self._video_cache:
            reader = self._video_cache.pop(path)
            self._video_cache[path] = reader
            return reader
        reader = _VideoReader(path)
        self._video_cache[path] = reader
        if len(self._video_cache) > self._max_cached_videos:
            _, old_reader = self._video_cache.popitem(last=False)
            old_reader.close()
        return reader
