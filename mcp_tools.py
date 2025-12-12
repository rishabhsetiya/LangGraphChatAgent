import httpx
import itertools
import json
from pydantic import BaseModel, create_model, Field
from langchain_core.tools import StructuredTool
from typing import Any, Dict, List, Callable

# --- MCP Tool Wrapper ---

def _create_mcp_tool(client: 'MCPClient', tool_info: Dict[str, Any]) -> StructuredTool:
    """Creates a LangChain-compatible StructuredTool from MCP tool information."""
    name = tool_info["name"]
    description = tool_info["description"]
    mcp_schema = tool_info["inputSchema"]

    # Extract required fields for correct handling of defaults/optional fields
    required_args = mcp_schema.get("required", [])

    # 1. Dynamically create the Pydantic BaseModel
    fields = {}
    for prop_name, prop_data in mcp_schema.get("properties", {}).items():

        # Determine Python type
        py_type = int if prop_data.get("type") == "integer" else str

        # Determine field definition (including default/description)
        field_definition = Field(description=prop_data.get("description", ""))

        prop_default = prop_data.get("default")

        if prop_name in required_args:
            # Required argument: Type, Field()
            fields[prop_name] = (py_type, field_definition)
        elif prop_default is not None:
            # Optional with default: Type, Field(default=...)
            fields[prop_name] = (py_type, Field(default=prop_default, description=prop_data.get("description", "")))
        else:
            # Optional without default: Type | None, None (or Ellipsis, depending on pydantic version)
            # Simplest form for optional fields:
            fields[prop_name] = (py_type | None, None)

            # Create the Pydantic class dynamically
    DynamicArgsModel = create_model(f"{name}_args", **fields, __base__=BaseModel)

    # 2. Define the execution function
    def mcp_executor(**kwargs: Any) -> str:
        """Dynamically generated tool executor for MCP."""
        return client.call_tool(name, kwargs)

    # 3. Use StructuredTool.from_function with the explicit schema
    return StructuredTool.from_function(
        func=mcp_executor,
        name=name,
        description=description,
        args_schema=DynamicArgsModel,
        handle_tool_errors=True
    )


# --- load_mcp_tools function ---

def load_mcp_tools(base_url: str) -> List[Callable]:
    """
    Connects to the MCP server, lists tools, and returns a list of
    LangChain-compatible tool functions.

    NOTE: The client object needs to be managed properly (e.g., using a
    global or passed explicitly if the client is not thread-safe or needs
    connection pooling). For simplicity, we initialize it here.
    """
    print(f"Loading tools from MCP server at {base_url}...")
    client = MCPClient(base_url)

    try:
        tool_list = client.list_tools()
        langchain_tools = [_create_mcp_tool(client, info) for info in tool_list]
        print(f"Successfully loaded {len(langchain_tools)} MCP tools.")
        return langchain_tools
    except Exception as e:
        print(f"WARNING: Could not load tools from MCP: {e}")
        return []
    finally:
        # NOTE: closing the client here would break the tool wrapper,
        # as it relies on the client. In a real app, the client must
        # be kept alive, e.g., by making it a singleton or managed resource.
        # For this example, we will let it leak or require the user to manage it.
        pass


# --- NEW/MODIFIED MCPClient class ---
class MCPClient:

    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/")
        self._id_counter = itertools.count(1)
        self.session_id = None  # Storage for the Mcp-Session-Id

        # Required headers for the Streamable MCP server
        required_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json"
        }

        # Initialize httpx client with the required headers
        self.client = httpx.Client(timeout=None, headers=required_headers)

        # 1. Perform the Session Initialization Handshake
        self._initialize_session()

    def _initialize_session(self):
        """Performs the MCP 'initialize' handshake."""
        print("Starting MCP session initialization...")

        # --- 1. Send 'initialize' request ---
        initialize_params = {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": True, "prompts": True, "resources": True,
                "logging": False, "elicitation": {}, "roots": {"listChanged": False}
            },
            "clientInfo": {"name": "python-client", "version": "1.0.0"}
        }

        # We need the full response to check headers, so we call the underlying post directly
        response = self._raw_request("initialize", initialize_params)

        # Save the Mcp-Session-Id from the response headers
        if "Mcp-Session-Id" in response.headers:
            self.session_id = response.headers["Mcp-Session-Id"]
            print(f"Session ID acquired: {self.session_id}")
        else:
            raise RuntimeError("MCP server did not return the required 'Mcp-Session-Id' header during initialization.")

        print("MCP session successfully initialized.")

    def _raw_request(self, method: str, params: Dict[str, Any] | None = None) -> httpx.Response:
        """Sends the request using the raw httpx client, returns the response object."""
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
        }

        # Conditional inclusion of params (prevents 400 error)
        if params is not None and params != {}:
            payload["params"] = params

        # Add the Mcp-Session-Id header if it exists
        headers = {}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        # Use the internal client which already has Content-Type and Accept headers set
        response = self.client.post(self.url, json=payload, headers=headers)
        response.raise_for_status()  # Check for 400, 500 errors
        return response

    def _request(self, method: str, params: Dict[str, Any] | None = None):
        """Sends the request, processes the response, and returns the result."""
        response = self._raw_request(method, params)

        # 1. Check for 202 Accepted (standard)
        if response.status_code == 202:
            return None

            # 2. Try standard JSON parsing first (in case the server switches modes)
        try:
            data = response.json()
        except Exception:
            # 3. If standard JSON fails, handle SSE format (id/event/data)
            # We look for the line that starts with "data:"
            raw_text = response.text
            json_str = None

            for line in raw_text.splitlines():
                if line.startswith("data:"):
                    # Remove "data:" prefix and strip whitespace
                    json_str = line[5:].strip()
                    break

            if json_str:
                data = json.loads(json_str)
            else:
                raise RuntimeError(f"Could not parse response: {raw_text}")

        # Standard Error Handling
        if "error" in data:
            raise RuntimeError(data["error"])

        return data.get("result", data)

    # -------- MCP API (Uses the updated _request method) --------

    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools exposed by the MCP server"""
        # This call now automatically includes the Session ID
        response = self._request("tools/list")["tools"]
        return response

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool and return its text output"""
        result = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments
            }
        )
        # Spring AI MCP returns content blocks
        if "content" in result and len(result["content"]) > 0:
            block = result["content"][0]
            return block.get("text") or str(block)
        return str(result)

    def close(self):
        self.client.close()