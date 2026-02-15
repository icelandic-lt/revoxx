"""Tests for tk_compat utilities."""

import unittest
from unittest.mock import Mock, patch


class TestDeferredWarning(unittest.TestCase):
    """Tests for deferred_warning()."""

    def test_calls_after(self):
        """deferred_warning schedules a callback via parent.after(0, ...)."""
        from revoxx.utils.tk_compat import deferred_warning

        parent = Mock()

        deferred_warning(parent, "Title", "Message")

        parent.after.assert_called_once()
        args = parent.after.call_args[0]
        self.assertEqual(args[0], 0)
        self.assertTrue(callable(args[1]))

    @patch("revoxx.utils.tk_compat.messagebox")
    def test_callback_shows_messagebox(self, mock_messagebox):
        """The scheduled callback calls messagebox.showwarning with correct args."""
        from revoxx.utils.tk_compat import deferred_warning

        parent = Mock()

        deferred_warning(parent, "Warn Title", "Warn body")

        callback = parent.after.call_args[0][1]
        callback()

        mock_messagebox.showwarning.assert_called_once_with(
            "Warn Title", "Warn body", parent=parent
        )


if __name__ == "__main__":
    unittest.main()
