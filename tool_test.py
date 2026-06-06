from src.tools import ToolRegistry, default_tools


reg = ToolRegistry("./test")


default_tools(reg)


print(
    reg.build_tool_context(
        "Build me a cli tool to create passkeys and store it on disk"
    )
)
