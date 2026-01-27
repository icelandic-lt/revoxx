"""Undo/redo stack for audio editing operations.

This module provides the UndoStack class that manages undo and redo
operations using the command pattern.
"""

from typing import TYPE_CHECKING, List, Optional

from .edit_commands import EditCommand

if TYPE_CHECKING:
    from ..files.file_manager import FileManager


class UndoStack:
    """Manages undo and redo stacks for edit commands.

    Uses two stacks:
    - Undo stack: Commands that can be undone (most recent on top)
    - Redo stack: Commands that were undone and can be redone

    When a new command is pushed, the redo stack is cleared.
    """

    def __init__(self, max_size: int = 50):
        """Initialize the undo stack.

        Args:
            max_size: Maximum number of commands to keep in the undo stack.
                      Older commands are discarded when the limit is reached.
        """
        self._max_size = max_size
        self._undo_stack: List[EditCommand] = []
        self._redo_stack: List[EditCommand] = []

    def push(self, command: EditCommand) -> None:
        """Push a new command onto the undo stack.

        This clears the redo stack since a new action invalidates
        any previously undone commands.

        Args:
            command: The command that was just executed
        """
        self._undo_stack.append(command)
        self._redo_stack.clear()

        if len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)

    def undo(self, file_manager: "FileManager") -> Optional[EditCommand]:
        """Undo the most recent command.

        Pops the command from the undo stack, executes its inverse,
        and pushes the original command to the redo stack.

        Args:
            file_manager: FileManager instance for audio I/O

        Returns:
            The command that was undone, or None if undo failed or stack is empty
        """
        if not self._undo_stack:
            return None

        command = self._undo_stack.pop()
        inverse_command = command.inverse()

        if inverse_command.execute(file_manager):
            self._redo_stack.append(command)
            return command

        self._undo_stack.append(command)
        return None

    def redo(self, file_manager: "FileManager") -> Optional[EditCommand]:
        """Redo the most recently undone command.

        Pops the command from the redo stack, re-executes it,
        and pushes it back to the undo stack.

        Args:
            file_manager: FileManager instance for audio I/O

        Returns:
            The command that was redone, or None if redo failed or stack is empty
        """
        if not self._redo_stack:
            return None

        command = self._redo_stack.pop()

        if command.execute(file_manager):
            self._undo_stack.append(command)
            return command

        self._redo_stack.append(command)
        return None

    def can_undo(self) -> bool:
        """Check if there are commands that can be undone.

        Returns:
            True if undo is available, False otherwise
        """
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if there are commands that can be redone.

        Returns:
            True if redo is available, False otherwise
        """
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        """Clear both undo and redo stacks.

        Call this when switching to a different audio file.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()

    def peek_undo(self) -> Optional[EditCommand]:
        """Get the next command that would be undone without removing it.

        Returns:
            The command at the top of the undo stack, or None if empty
        """
        if self._undo_stack:
            return self._undo_stack[-1]
        return None

    def peek_redo(self) -> Optional[EditCommand]:
        """Get the next command that would be redone without removing it.

        Returns:
            The command at the top of the redo stack, or None if empty
        """
        if self._redo_stack:
            return self._redo_stack[-1]
        return None

    def undo_description(self) -> Optional[str]:
        """Get description of the next undo operation.

        Returns:
            Description string or None if undo stack is empty
        """
        cmd = self.peek_undo()
        return cmd.description() if cmd else None

    def redo_description(self) -> Optional[str]:
        """Get description of the next redo operation.

        Returns:
            Description string or None if redo stack is empty
        """
        cmd = self.peek_redo()
        return cmd.description() if cmd else None

    @property
    def undo_count(self) -> int:
        """Get the number of commands in the undo stack."""
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        """Get the number of commands in the redo stack."""
        return len(self._redo_stack)
