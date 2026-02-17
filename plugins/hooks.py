"""
Plugin hook registry for FRAMEWORM.

Provides lifecycle hooks that plugins can subscribe to, enabling
zero-modification extensibility across the training loop and framework.

Usage:
    from plugins.hooks import HookRegistry, hook

    registry = HookRegistry()

    # Register a hook function
    @registry.register("on_epoch_end")
    def my_hook(epoch, metrics, **kwargs):
        print(f"Epoch {epoch} done: {metrics}")

    # Or use the CallbackHook base class
    class MyPlugin(CallbackHook):
        def on_epoch_end(self, epoch, metrics, **kwargs):
            print(f"Epoch {epoch} done: {metrics}")

    registry.register_object(MyPlugin(priority=10))

    # Fire hooks from Trainer
    registry.call("on_epoch_end", epoch=5, metrics={"loss": 0.4})
"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# All supported lifecycle hook names
SUPPORTED_HOOKS = [
    # Training lifecycle
    "on_train_begin",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    # Batch lifecycle
    "on_batch_begin",
    "on_batch_end",
    # Validation lifecycle
    "on_validation_begin",
    "on_validation_end",
    # Checkpoint lifecycle
    "on_checkpoint_save",
    "on_checkpoint_load",
    # Model lifecycle
    "on_model_export",
    "on_model_register",
    # Search lifecycle
    "on_trial_begin",
    "on_trial_end",
    "on_search_complete",
    # Error lifecycle
    "on_error",
]


class HookEntry:
    """Internal record for a registered hook function."""

    def __init__(self, fn: Callable, priority: int, name: str, enabled: bool = True):
        self.fn = fn
        self.priority = priority
        self.name = name
        self.enabled = enabled

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"HookEntry(name={self.name!r}, priority={self.priority}, {status})"


class HookRegistry:
    """
    Central registry for lifecycle hooks.

    Hooks are called in ascending priority order (lower number = called first).
    Default priority is 50. Use lower values (e.g. 10) to run before others,
    higher values (e.g. 90) to run after.

    Thread-safety: Hook registration is not thread-safe. Register all hooks
    before starting training. Calling hooks IS safe from multiple threads
    as long as the hook functions themselves are thread-safe.

    Example:
        >>> registry = HookRegistry()

        >>> @registry.register("on_epoch_end")
        ... def log_metrics(epoch, metrics, **kwargs):
        ...     print(metrics)

        >>> registry.call("on_epoch_end", epoch=1, metrics={"loss": 0.5})
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, raise ValueError when calling unknown hook names.
                    If False (default), log a warning and continue.
        """
        self._hooks: Dict[str, List[HookEntry]] = {name: [] for name in SUPPORTED_HOOKS}
        self._strict = strict
        self._enabled = True

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        hook_name: str,
        priority: int = 50,
    ) -> Callable:
        """
        Decorator to register a function for a hook.

        Args:
            hook_name: One of the SUPPORTED_HOOKS names.
            priority:  Execution order (lower = earlier). Default 50.

        Returns:
            The original function, unchanged.

        Example:
            >>> @registry.register("on_train_begin")
            ... def setup(config, **kwargs):
            ...     print("Training starting!")
        """
        def decorator(fn: Callable) -> Callable:
            self._add(hook_name, fn, priority, name=fn.__qualname__)
            return fn
        return decorator

    def register_fn(
        self,
        hook_name: str,
        fn: Callable,
        priority: int = 50,
    ) -> None:
        """
        Register a function directly (non-decorator form).

        Args:
            hook_name: One of the SUPPORTED_HOOKS names.
            fn:        Callable to register.
            priority:  Execution order (lower = earlier). Default 50.
        """
        self._add(hook_name, fn, priority, name=fn.__qualname__)

    def register_object(self, obj: "CallbackHook") -> None:
        """
        Register all hook methods from a CallbackHook instance.

        Any method whose name matches a supported hook name is registered
        automatically.

        Args:
            obj: Instance of CallbackHook (or any object with matching methods).
        """
        registered = 0
        for hook_name in SUPPORTED_HOOKS:
            method = getattr(obj, hook_name, None)
            if method is not None and callable(method):
                self._add(hook_name, method, obj.priority, name=type(obj).__name__)
                registered += 1
        if registered == 0:
            warnings.warn(
                f"{type(obj).__name__} has no methods matching any supported hook names.",
                UserWarning,
            )

    def unregister(self, hook_name: str, fn: Callable) -> bool:
        """
        Remove a specific function from a hook.

        Returns:
            True if the function was found and removed, False otherwise.
        """
        if hook_name not in self._hooks:
            return False
        before = len(self._hooks[hook_name])
        self._hooks[hook_name] = [e for e in self._hooks[hook_name] if e.fn is not fn]
        return len(self._hooks[hook_name]) < before

    def clear(self, hook_name: Optional[str] = None) -> None:
        """
        Remove all registered hooks.

        Args:
            hook_name: If given, clear only that hook. Otherwise clear all.
        """
        if hook_name is not None:
            if hook_name in self._hooks:
                self._hooks[hook_name] = []
        else:
            for name in self._hooks:
                self._hooks[name] = []

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(self, hook_name: Optional[str] = None) -> None:
        """Enable hooks globally or for a specific hook name."""
        if hook_name is None:
            self._enabled = True
        else:
            for entry in self._hooks.get(hook_name, []):
                entry.enabled = True

    def disable(self, hook_name: Optional[str] = None) -> None:
        """Disable hooks globally or for a specific hook name (useful for debugging)."""
        if hook_name is None:
            self._enabled = False
        else:
            for entry in self._hooks.get(hook_name, []):
                entry.enabled = False

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    def call(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """
        Fire all registered functions for a hook.

        Functions are called in ascending priority order.
        Exceptions in individual hooks are caught and logged so one bad
        hook cannot break training.

        Args:
            hook_name: Hook to fire.
            **kwargs:  Passed through to every registered function.

        Returns:
            List of return values from each hook function (None values excluded).
        """
        if not self._enabled:
            return []

        if hook_name not in self._hooks:
            msg = f"Unknown hook: {hook_name!r}. Supported: {SUPPORTED_HOOKS}"
            if self._strict:
                raise ValueError(msg)
            logger.warning(msg)
            return []

        results = []
        for entry in self._hooks[hook_name]:  # already sorted on insert
            if not entry.enabled:
                continue
            try:
                result = entry.fn(**kwargs)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.error(
                    "Hook %r (registered by %r) raised an exception: %s",
                    hook_name,
                    entry.name,
                    exc,
                    exc_info=True,
                )
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_hooks(self, hook_name: Optional[str] = None) -> Dict[str, List[HookEntry]]:
        """Return registered hooks, optionally filtered by name."""
        if hook_name is not None:
            return {hook_name: self._hooks.get(hook_name, [])}
        return {k: v for k, v in self._hooks.items() if v}

    def summary(self) -> None:
        """Print a human-readable summary of all registered hooks."""
        active = {k: v for k, v in self._hooks.items() if v}
        if not active:
            print("HookRegistry: no hooks registered.")
            return
        print(f"HookRegistry ({'enabled' if self._enabled else 'DISABLED'}):")
        for hook_name, entries in active.items():
            print(f"  {hook_name}:")
            for e in entries:
                status = "✓" if e.enabled else "✗"
                print(f"    [{status}] priority={e.priority:3d}  {e.name}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, hook_name: str, fn: Callable, priority: int, name: str) -> None:
        if hook_name not in self._hooks:
            msg = f"Unknown hook: {hook_name!r}. Supported: {SUPPORTED_HOOKS}"
            if self._strict:
                raise ValueError(msg)
            logger.warning(msg)
            self._hooks[hook_name] = []

        entry = HookEntry(fn=fn, priority=priority, name=name)
        self._hooks[hook_name].append(entry)
        # Keep sorted by priority (stable sort preserves registration order for ties)
        self._hooks[hook_name].sort(key=lambda e: e.priority)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._hooks.values())
        return f"HookRegistry(total_hooks={total}, enabled={self._enabled})"


# ---------------------------------------------------------------------------
# CallbackHook base class
# ---------------------------------------------------------------------------

class CallbackHook:
    """
    Base class for object-oriented plugins.

    Subclass this and override any hook methods you need.
    Pass the instance to ``registry.register_object()``.

    Priority controls execution order across all registered objects.
    Lower number = called first. Default is 50.

    Example:
        >>> class WandBPlugin(CallbackHook):
        ...     def __init__(self):
        ...         super().__init__(priority=20)
        ...
        ...     def on_train_begin(self, config, **kwargs):
        ...         import wandb
        ...         wandb.init(config=config)
        ...
        ...     def on_epoch_end(self, epoch, metrics, **kwargs):
        ...         wandb.log(metrics, step=epoch)
        ...
        ...     def on_train_end(self, **kwargs):
        ...         wandb.finish()
    """

    def __init__(self, priority: int = 50):
        self.priority = priority

    # Training lifecycle stubs (override as needed)
    def on_train_begin(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_validation_begin(self, **kwargs): pass
    def on_validation_end(self, **kwargs): pass
    def on_checkpoint_save(self, **kwargs): pass
    def on_checkpoint_load(self, **kwargs): pass
    def on_model_export(self, **kwargs): pass
    def on_model_register(self, **kwargs): pass
    def on_trial_begin(self, **kwargs): pass
    def on_trial_end(self, **kwargs): pass
    def on_search_complete(self, **kwargs): pass
    def on_error(self, **kwargs): pass

    def __repr__(self):
        return f"{type(self).__name__}(priority={self.priority})"


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry = HookRegistry()


def get_default_registry() -> HookRegistry:
    """Return the module-level default HookRegistry."""
    return _default_registry


def hook(hook_name: str, priority: int = 50) -> Callable:
    """
    Convenience decorator using the default registry.

    Example:
        >>> from plugins.hooks import hook

        >>> @hook("on_epoch_end")
        ... def my_fn(epoch, metrics, **kwargs):
        ...     print(epoch, metrics)
    """
    return _default_registry.register(hook_name, priority=priority)