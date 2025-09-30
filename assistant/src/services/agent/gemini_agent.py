import operator
from typing import TypedDict, Annotated, AsyncIterator, List

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from loguru import logger

from assistant.src.config import Settings
from assistant.src.core.interfaces import Agent
from assistant.src.core.types import ChatMessage, Role, ToolCall
from assistant.src.services.tools.registry import ToolRegistry
from assistant.src.core.exceptions import AgentError


class AgentState(TypedDict):
    """
    Represents the state of our agent. It's a dictionary that holds a list of messages.
    The `operator.add` annotation tells LangGraph to append new messages to this list
    rather than replacing them.
    """
    messages: Annotated[list[AnyMessage], operator.add]


class GeminiAgent(Agent):
    """
    The "brain" of the assistant. It uses Gemini and LangGraph to manage conversation
    state, process user input, and decide when to use tools.
    """

    def __init__(self, settings: Settings, tool_registry: ToolRegistry):
        """
        Initializes the agent, LLM, tools, and the computational graph.

        Args:
            settings: The application settings object.
            tool_registry: The registry containing all available tools.
        """
        self.settings = settings
        self.tool_registry = tool_registry
        self.tools = self.tool_registry.get_all_tools_langchain()

        logger.info(f"Initializing GeminiAgent with model: {self.settings.llm_model}")

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                google_api_key=self.settings.gemini_api_key,
                temperature=self.settings.llm_temperature,
                convert_system_message_to_human=True, # Recommended for Gemini
            )
            # Bind the tools to the LLM so it knows their signatures.
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception as e:
            logger.opt(exception=e).critical("Failed to initialize ChatGoogleGenerativeAI.")
            raise AgentError("Could not initialize the Gemini LLM.") from e

        self.graph = self._build_graph()
        logger.success(f"GeminiAgent initialized with {len(self.tools)} tools.")

    def _build_graph(self):
        """Builds the state graph that controls the agent's execution flow."""
        graph = StateGraph(AgentState)

        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", self._call_tool)

        graph.set_entry_point("llm")

        graph.add_conditional_edges(
            "llm",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "llm")

        return graph.compile()

    def _should_continue(self, state: AgentState) -> str:
        """Determines the next step after the LLM has been called."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        return "end"

    async def _call_llm(self, state: AgentState) -> dict:
        """Invokes the LLM with the current conversation history."""
        response = await self.llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    async def _call_tool(self, state: AgentState) -> dict:
        """Executes the tool(s) requested by the LLM."""
        last_message = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info(f"Using tool: {tool_name} with args: {tool_args}")
            
            tool_result_message = await self.tool_registry.call_tool(
                tool_name, tool_args, tool_call["id"]
            )
            tool_messages.append(tool_result_message)
        
        return {"messages": tool_messages}

    def _convert_to_langchain_history(self, history: List[ChatMessage]) -> List[AnyMessage]:
        """Converts our internal ChatMessage format to LangChain's format."""
        lc_history = []
        for msg in history:
            if msg.role == Role.USER:
                lc_history.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                # Reconstruct AIMessage with potential tool calls
                ai_msg = AIMessage(content=msg.content or "")
                if msg.tool_calls:
                    ai_msg.tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": tc.function.arguments,
                        }
                        for tc in msg.tool_calls
                    ]
                lc_history.append(ai_msg)
        return lc_history
    
    async def get_response(
        self,
        history: List[ChatMessage],
        last_user_message: str,
    ) -> AsyncIterator[ChatMessage]:
        """
        Processes user input and generates a stream of response messages,
        handling tool calls transparently.
        """
        system_prompt = (
            "You are a helpful and concise voice assistant. "
            "Your responses will be spoken, so do not use markdown formatting "
            "like _italics_, **bold**, or `code blocks`."
        )

        initial_messages = [
            SystemMessage(content=system_prompt),
            *self._convert_to_langchain_history(history),
            HumanMessage(content=last_user_message),
        ]
        
        final_state: AgentState | None = None
        
        async for chunk in self.graph.astream({"messages": initial_messages}):
            if END in chunk:
                final_state = chunk[END]
                break

        if not final_state or not final_state.get("messages"):
             logger.error("Agent graph finished with no final state or messages.")
             yield ChatMessage(role=Role.ASSISTANT, content="I'm sorry, but something went wrong.")
             return

        final_message = final_state["messages"][-1]

        # Stream the final text response token by token
        async for token in self.llm.astream(final_state["messages"]):
             yield ChatMessage(role=Role.ASSISTANT, content=token.content)