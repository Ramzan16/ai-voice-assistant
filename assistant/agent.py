import operator
from typing import TypedDict, Annotated, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from assistant.tools import load_all_tools
import logging

class AgentState(TypedDict):
    # ... existing AgentState class ...
    messages: Annotated[list[AnyMessage], operator.add]

class LLMProcessor:
    """
    The "brain" of the assistant. It uses an LLM and LangGraph to manage conversation
    state, process user input, and decide when to use tools.
    Now supports streaming output.
    """
    def __init__(self):
        # Initialize the LLM and the tools
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, google_api_key=config.GOOGLE_API_KEY)
        self.tools = load_all_tools()
        # Bind the tools to the LLM so it knows their signatures
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # Build the graph that defines the agent's logic
        self.graph = self._build_graph()
        logging.info(f"LLM Processor initialized with {len(self.tools)} tools.")

    def _build_graph(self):
        # ... existing _build_graph() code ...
        graph = StateGraph(AgentState)

        # Define the two main nodes in our graph
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", self._call_tool)

        # The entry point is the LLM
        graph.set_entry_point("llm")

        # Define the conditional logic for routing
        graph.add_conditional_edges(
            "llm",
            self._should_continue,
            {
                "continue": "tools",  # If a tool call is present, go to the tools node
                "end": END           # Otherwise, end the conversation turn
            }
        )

        # After calling a tool, the flow always goes back to the LLM
        graph.add_edge("tools", "llm")

        # Compile the graph into a runnable object
        return graph.compile()


    def _should_continue(self, state: AgentState):
        # ... existing _should_continue() code ...
        if state["messages"][-1].tool_calls:
            return "continue"
        return "end"

    def _call_llm(self, state: AgentState):
        # ... existing _call_llm() code ...
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}


    def _call_tool(self, state: AgentState):
        # ... existing _call_tool() code ...
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            # Find the corresponding tool function from our loaded tools
            tool_to_call = next(t for t in self.tools if t.name == tool_name)
            try:
                # Invoke the tool and get the output
                output = tool_to_call.invoke(tool_call["args"])
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                tool_messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"])
                )
        return {"messages": tool_messages}

    def process(self, text: str) -> Generator[str, None, None]:
        """
        Processes the user's input and streams the response from the LLM.
        Note: Tool usage is currently not streamed, the graph will run to completion
        and then the final response will be streamed. A more complex graph would be
        needed for streaming intermediate tool steps.

        Args:
            text: The transcribed text from the user.

        Yields:
            A generator of text chunks from the LLM's final response.
        """
        system_message = SystemMessage(content="You are a helpful voice assistant. Be concise.")
        initial_messages = [system_message, HumanMessage(content=text)]
        
        # Use .stream() instead of .invoke()
        stream = self.graph.stream({"messages": initial_messages})
        
        final_response_streamed = False
        for chunk in stream:
            # The stream yields the state of the graph at each step.
            # We are interested in the final LLM response.
            if "llm" in chunk and not final_response_streamed:
                # Get the last message from the 'llm' node output
                last_message = chunk["llm"]["messages"][-1]
                # If it's a tool call, we wait. If it's a final answer, we stream it.
                if not last_message.tool_calls:
                    final_response_streamed = True
                    # This is the final response stream, yield its content chunk by chunk
                    for token in self.llm.stream(last_message.content):
                        yield token.content