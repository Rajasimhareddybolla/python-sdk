from unittest.mock import Mock

import pytest
from mcp_types import CallToolResult, TextContent

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.prompts.base import Prompt
from mcp.server.mcpserver.resources.base import Resource
from mcp.server.mcpserver.resources.templates import ResourceTemplate

pytestmark = pytest.mark.anyio


async def test_positional_only_tool_execution() -> None:
    """Test that tools with positional-only parameters execute successfully."""
    mcp = MCPServer("test")

    @mcp.tool()
    def echo_sync(x: str, /) -> str:
        return f"got:{x}"

    @mcp.tool()
    async def echo_async(x: str, /, y: int = 10, *, z: str = "default") -> str:
        return f"x={x},y={y},z={z}"

    ctx = Mock()

    res_sync = await mcp.call_tool("echo_sync", {"x": "hello"}, ctx)
    assert isinstance(res_sync, CallToolResult)
    assert not res_sync.is_error
    assert res_sync.content == [TextContent(type="text", text="got:hello")]
    assert res_sync.structured_content == {"result": "got:hello"}

    res_async = await mcp.call_tool("echo_async", {"x": "foo", "y": 20, "z": "custom"}, ctx)
    assert isinstance(res_async, CallToolResult)
    assert not res_async.is_error
    assert res_async.content == [TextContent(type="text", text="x=foo,y=20,z=custom")]


async def test_positional_only_prompt_render() -> None:
    """Test that prompts with positional-only parameters render correctly."""

    def greet_sync(name: str, /) -> str:
        return f"hello {name}"

    async def greet_async(name: str, /, title: str = "Mr") -> str:
        return f"hello {title} {name}"

    prompt_sync = Prompt.from_function(greet_sync)
    prompt_async = Prompt.from_function(greet_async)
    ctx = Mock()

    msgs_sync = await prompt_sync.render({"name": "Alice"}, ctx)
    assert isinstance(msgs_sync, list)
    assert len(msgs_sync) == 1
    content_sync = msgs_sync[0].content
    assert isinstance(content_sync, TextContent)
    assert content_sync.text == "hello Alice"

    msgs_async = await prompt_async.render({"name": "Bob", "title": "Dr"}, ctx)
    assert isinstance(msgs_async, list)
    assert len(msgs_async) == 1
    content_async = msgs_async[0].content
    assert isinstance(content_async, TextContent)
    assert content_async.text == "hello Dr Bob"


async def test_positional_only_resource_template() -> None:
    """Test that resource templates with positional-only parameters create resources correctly."""

    def read_sync(item: str, /) -> str:
        return f"data:{item}"

    async def read_async(item: str, /) -> str:
        return f"async_data:{item}"

    template_sync = ResourceTemplate.from_function(read_sync, uri_template="test://{item}")
    template_async = ResourceTemplate.from_function(read_async, uri_template="async://{item}")
    ctx = Mock()

    res_sync = await template_sync.create_resource("test://alpha", {"item": "alpha"}, ctx)
    assert isinstance(res_sync, Resource)
    content_sync = await res_sync.read()
    assert content_sync == "data:alpha"

    res_async = await template_async.create_resource("async://beta", {"item": "beta"}, ctx)
    assert isinstance(res_async, Resource)
    content_async = await res_async.read()
    assert content_async == "async_data:beta"
