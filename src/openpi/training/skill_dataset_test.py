import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import polars as pl

from openpi.training.skill_dataset import SkillSequenceDataset


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def test_skill_sequence_dataset(tmp_path: Path) -> None:
    dataset_info = {
        "chunks_size": 10000,
        "splits": {"train": "0:1"},
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
    }
    info_path = tmp_path / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info))

    # Create Parquet data for a single episode with 5 frames.
    data_dir = tmp_path / "data" / "task-0000"
    data_dir.mkdir(parents=True)
    episode_actions = []
    episode_states = []
    for frame_idx in range(5):
        episode_actions.append(list(np.full(3, frame_idx, dtype=np.float32)))
        episode_states.append(list(np.arange(4, dtype=np.float32) + frame_idx))
    df = pl.DataFrame(
        {
            "action": pl.Series(episode_actions, dtype=pl.List(pl.Float32)),
            "observation.state": pl.Series(episode_states, dtype=pl.List(pl.Float32)),
        }
    )
    data_path = data_dir / "episode_00000000.parquet"
    df.write_parquet(data_path)

    # Create annotation with two skills.
    annotations_dir = tmp_path / "annotations" / "task-0000"
    annotations_dir.mkdir(parents=True)
    annotation = {
        "skill_annotation": [
            {
                "skill_idx": 0,
                "skill_description": ["move to"],
                "object_id": [["object_a"]],
                "manipulating_object_id": [],
                "memory_prefix": [],
                "spatial_prefix": [],
                "skill_type": ["navigation"],
                "frame_duration": [0, 3],
            },
            {
                "skill_idx": 1,
                "skill_description": ["place"],
                "object_id": [["object_b"]],
                "manipulating_object_id": [],
                "memory_prefix": [],
                "spatial_prefix": [],
                "skill_type": ["placement"],
                "frame_duration": [3, 5],
            },
        ]
    }
    (annotations_dir / "episode_00000000.json").write_text(json.dumps(annotation))

    # Create videos for each camera.
    camera_keys = [
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    ]
    for idx, feature_key in enumerate(camera_keys):
        frames = [
            np.full((32, 32, 3), fill_value=(frame_idx + idx) * 10, dtype=np.uint8)
            for frame_idx in range(5)
        ]
        video_dir = tmp_path / "videos" / "task-0000" / feature_key
        video_dir.mkdir(parents=True)
        _write_video(video_dir / "episode_00000000.mp4", frames)

    dataset = SkillSequenceDataset(
        root_dir=tmp_path,
        dataset_info_path=info_path,
        split="train",
        action_horizon=4,
        image_feature_map={
            "observation.images.rgb.head": "base_0_rgb",
            "observation.images.rgb.left_wrist": "left_wrist_0_rgb",
            "observation.images.rgb.right_wrist": "right_wrist_0_rgb",
        },
        state_key="observation.state",
        action_key="action",
        prompt_template="{description} {objects}",
        frame_selection="middle",
        max_cached_episodes=2,
        max_cached_videos=4,
    )

    assert len(dataset) == 2

    sample0 = dataset[0]
    assert set(sample0["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for image in sample0["image"].values():
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
    for mask in sample0["image_mask"].values():
        assert mask.dtype == np.bool_
        assert mask.shape == ()
        assert bool(mask)
    assert sample0["actions"].shape == (4, 3)
    assert sample0["state"].shape == (4,)
    np.testing.assert_allclose(sample0["actions"][-1], np.full(3, 2, dtype=np.float32))
    assert sample0["prompt"] == "move to object_a"

    sample1 = dataset[1]
    assert sample1["prompt"] == "place object_b"
    np.testing.assert_allclose(sample1["actions"][0], np.full(3, 3, dtype=np.float32))
