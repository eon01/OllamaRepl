# repl.py (pass 5: real summarization via LangChain)
"""A chat REPL that compresses old conversation history instead of dropping it.

We keep pass 4's trim_history function as a fallback layer below the
middleware, but in practice SummarizationMiddleware does the heavy
lifting: it intercepts every model call, counts tokens across the
conversation, and when the total crosses our trigger threshold it
asks the model itself to summarize the older messages.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_HOST, OLLAMA_MODEL


def read_input() -> str:
    """Read lines from stdin until the user submits an empty line."""
    lines: list[str] = []
    prompt = "> "
    while True:
        line = input(prompt)
        if line == "":
            break
        lines.append(line)
        prompt = "  "
    return "\n".join(lines)


def main() -> None:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    # The "summarizer" model. In production setups people often pick a
    # smaller or faster model here, since summarization is a simpler task
    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    # create_agent builds an executable agent around the worker model.
    # The middleware list runs in order; ours has just one entry, which
    # fires before each model call and decides whether to summarize.
    #
    # trigger=("tokens", 2000) means: if the conversation reaches 2000
    # tokens, summarize older messages before the next call.

    # keep=("messages", 6) means: preserve the most recent 6 messages
    # verbatim. Everything older becomes a summary.
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                trigger=("tokens", 2000),
                keep=("messages", 6),
            ),
        ],
    )

    print(f"Chatting with {OLLAMA_MODEL} (with summarization).")
    print("Hit Enter on an empty line to send. Type /bye to exit.")

    messages: list = []

    while True:
        user = read_input()
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append(HumanMessage(content=user))

        # stream_mode="messages" yields (chunk, metadata) tuples where
        # chunk.content is just the new tokens, not the cumulative reply.
        full_reply = ""
        for chunk, _ in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content

        print()
        messages.append(AIMessage(content=full_reply))


if __name__ == "__main__":
    main()
