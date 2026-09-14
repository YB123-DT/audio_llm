import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_ravdess_manifest import build_rows, validate_rows
from scripts.slam_omni_diagnostics import format_prompt


class BuildRavdessManifestTests(unittest.TestCase):
    def test_prompt_uses_official_slam_omni_system_wrapper(self) -> None:
        self.assertEqual(
            format_prompt("Listen to the speech."),
            "<SYSTEM>: Listen to the speech.\n ",
        )

    def test_strict_pairing_ignores_song_and_other_emotions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "audio_speech_actors_01-24" / "Actor_01"
            actor.mkdir(parents=True)
            names = [
                "03-01-03-01-01-01-01.wav",  # happy speech
                "03-01-04-01-01-01-01.wav",  # sad speech
                "03-02-03-01-01-01-01.wav",  # happy song
                "03-01-05-01-01-01-01.wav",  # angry speech
                "03-01-03-02-01-01-01.wav",  # other intensity
            ]
            for name in names:
                (actor / name).touch()
            rows = build_rows(root, "01")
            validate_rows(rows, root, "01")
            self.assertEqual([row["emotion"] for row in rows], ["happy", "sad"])
            self.assertEqual(rows[0]["actor"], "01")
            self.assertEqual(rows[0]["intensity"], "01")

    def test_missing_emotion_does_not_create_partial_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "Actor_02"
            actor.mkdir()
            (actor / "03-01-03-01-01-01-02.wav").touch()
            rows = build_rows(root, "01")
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
