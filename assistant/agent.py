import operator
from typing import TypedDict, Annotated, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from assistant.tools import load_all_tools
import logging
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class AgentState(TypedDict):
    """
    Represents the state of our agent. It's a dictionary that holds a list of messages.
    The `operator.add` annotation tells LangGraph to append new messages to this list
    rather than replacing it.
    """
    messages: Annotated[list[AnyMessage], operator.add]

class LLMProcessor:
    """
    The "brain" of the assistant. It uses an LLM and LangGraph to manage conversation
    state, process user input, and decide when to use tools.
    Now supports streaming output.
    """
    def __init__(self):
        """
        Initializes the LLM, loads the tools, binds them to the LLM, and
        builds the computational graph that defines the agent's logic.
        """
        # Temperature is set to 0.0 for more deterministic and factual responses.
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key, temperature=0.0)
        
        # Load the external tools the agent can use.
        self.tools = load_all_tools()
        
        # Bind the tools to the LLM. This allows the LLM to know the signatures
        # (name, description, arguments) of the available tools.
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph that defines the agent's control flow.
        self.graph = self._build_graph()
        logging.info(f"LLM Processor initialized with {len(self.tools)} tools.")

    def _build_graph(self):
        """
        Builds the state graph that controls the agent's execution flow.
        """
        graph = StateGraph(AgentState)

        # Define the two main nodes in our graph: one for calling the LLM and one for executing tools.
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", self._call_tool)

        # The entry point for the graph is the 'llm' node. Every new input starts here.
        graph.set_entry_point("llm")

        # Define the conditional logic for routing after the LLM has been called.
        graph.add_conditional_edges(
            "llm",
            self._should_continue,
            {
                "continue": "tools",  # If the LLM decided to use a tool, go to the 'tools' node.
                "end": END           # Otherwise, if it generated a direct answer, end the process.
            }
        )

        # After a tool is executed, the flow always returns to the LLM to process the tool's output.
        graph.add_edge("tools", "llm")

        # Compile the graph into a runnable object.
        return graph.compile()

    def _should_continue(self, state: AgentState) -> str:
        """
        Determines the next step after the LLM has been called.
        
        Args:
            state: The current state of the agent.
            
        Returns:
            'continue' if the LLM requested a tool call, 'end' otherwise.
        """
        # Check the last message in the state. If it contains tool calls, we continue to the tools node.
        if state["messages"][-1].tool_calls:
            return "continue"
        # If there are no tool calls, it means the LLM has provided a final answer.
        return "end"

    def _call_llm(self, state: AgentState) -> dict:
        """
        Invokes the LLM with the current conversation history (state).
        
        Args:
            state: The current state of the agent.
            
        Returns:
            A dictionary containing the LLM's response message.
        """
        # The llm_with_tools object is invoked, which might return a regular message or a tool call.
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _call_tool(self, state: AgentState) -> dict:
        """
        Executes the tool(s) requested by the LLM.
        
        Args:
            state: The current state of the agent, where the last message is a tool call.
            
        Returns:
            A dictionary containing the output from the tool(s) as ToolMessage(s).
        """
        # The last message should contain one or more tool calls.
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            # Find the corresponding tool function from our loaded tools list.
            tool_to_call = next((t for t in self.tools if t.name == tool_name), None)
            
            if not tool_to_call:
                 tool_messages.append(
                    ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"])
                )
                 continue

            try:
                # Invoke the tool with the arguments provided by the LLM.
                output = tool_to_call.invoke(tool_call["args"])
                # Create a ToolMessage to append to the conversation history.
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}")
                tool_messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"])
                )
        # This list of tool messages will be added to the state.
        return {"messages": tool_messages}

    def process(self, text: str) -> Generator[str, None, None]:
        """
        Processes the user's input and streams the response from the LLM.
        
        Note: Tool usage is currently not streamed. The graph will run to completion
        before the final response is streamed. A more complex graph would be
        needed for streaming intermediate tool steps.

        Args:
            text: The transcribed text from the user.

        Yields:
            A generator of text chunks from the LLM's final response.
        """
        system_message = SystemMessage(content="You are a helpful voice assistant. Be concise.")
        initial_messages = [system_message, HumanMessage(content=text)]
        
        # Use .stream() to get a generator of intermediate steps from the graph.
        stream = self.graph.stream({"messages": initial_messages})
        
        final_response_streamed = False
        for chunk in stream:
            # The stream yields the state of the graph at each step (e.g., {"llm": ...} or {"tools": ...}).
            # We are interested only in the final LLM response.
            if "llm" in chunk and not final_response_streamed:
                # Get the last message from the 'llm' node's output.
                last_message = chunk["llm"]["messages"][-1]
                
                # If it's a tool call, we ignore it and wait. If it's a final answer, we stream it.
                if not last_message.tool_calls:
                    final_response_streamed = True
                    # This is the final response stream. We now stream its content token by token.
                    # We stream directly from the base LLM, not the one with tools,
                    # as we are just streaming the content of the final message.
                    for token in self.llm.stream(last_message.content):
                        yield token.content