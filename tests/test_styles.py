import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from ui.styles import get_botanical_css

class TestBotanicalStyles(unittest.TestCase):
    def test_get_botanical_css_tokens(self):
        css = get_botanical_css()
        self.assertIn(".gournet-card", css)
        self.assertIn(".status-pill", css)
        self.assertIn(".metric-badge", css)
        self.assertIn(".consensus-banner", css)
        self.assertIn(".divergence-banner", css)
        # Check botanical color tokens
        self.assertTrue("#0F5132" in css or "#166534" in css or "#10B981" in css)

    def test_dark_mode_support(self):
        css = get_botanical_css()
        self.assertIn("@media (prefers-color-scheme: dark)", css)
        self.assertIn('[data-theme="dark"]', css)
        self.assertIn("--header-subtext-color", css)
        self.assertIn("--status-pill-color", css)
        self.assertIn("--consensus-color", css)
        self.assertIn("--divergence-color", css)

if __name__ == "__main__":
    unittest.main()
