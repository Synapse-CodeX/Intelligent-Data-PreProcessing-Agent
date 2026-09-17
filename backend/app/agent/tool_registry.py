from typing import Any

from app.agent.tools import AgentTool


class ToolRegistry:
    """Registry for tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """Register a tool using its unique name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")

        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        """Retrieve a registered tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")

        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def execute(
        self,
        name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a registered tool by name."""
        tool = self.get(name)
        return tool.run(**kwargs)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
