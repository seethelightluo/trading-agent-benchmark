import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from agent.toolkit.get_news import GetNewsTool


class NewsVisibilityTests(unittest.TestCase):
    def test_visible_through_blocks_same_day_and_future_news(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news_dir = root / "news"
            news_dir.mkdir()
            (root / "date.json").write_text(json.dumps({
                "current_date": "2026-07-17",
                "visible_through": "2026-07-16",
            }))
            (news_dir / "SPX.json").write_text(json.dumps([
                {"publish_date": "2026-07-16 09:00:00", "title": "visible"},
                {"publish_date": "2026-07-17 09:00:00", "title": "same day hidden"},
                {"publish_date": "2026-07-18 09:00:00", "title": "future hidden"},
            ]))
            tool = GetNewsTool(str(news_dir), str(root / "date.json"))
            output = tool.get_implementation()("SPX", days=30)
            self.assertIn("visible", output)
            self.assertNotIn("same day hidden", output)
            self.assertNotIn("future hidden", output)
            self.assertIn("as of 2026-07-16", output)


if __name__ == "__main__":
    unittest.main()
