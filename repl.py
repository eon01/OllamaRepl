# repl.py (pass 3: streaming + multi-line input)
"""A chat REPL with streaming output and multi-line input.

Multi-line input: type your message across as many lines as you like,
then send it by hitting Enter on an empty line. Type /bye on its own
line to exit.
"""

from ollama import Client
from config import OLLAMA_HOST, OLLAMA_MODEL


def read_input() -> str:
    """Read lines from stdin until the user submits an empty line.

    The first line uses a '> ' prompt, continuation lines use '  ' so
    the user can see they're still inside the same message. Returns the
    joined message with newlines preserved.
    """
    lines: list[str] = []
    prompt = "> "
    while True:
        line = input(prompt)
        # Empty line means "I'm done, send it". This applies even on the
        # first line: hitting Enter immediately just gives an empty turn,
        # which the caller can ignore.
        if line == "":
            break
        lines.append(line)
        prompt = "  "
    return "\n".join(lines)


def main() -> None:
    # We bump the client timeout because long replies can easily exceed
    # the default. Setting it to None disables the timeout entirely; we
    # use a generous number instead so a truly stuck server still fails.
    client = Client(host=OLLAMA_HOST, timeout=300)

    print(f"Chatting with {OLLAMA_MODEL}.")
    print("Hit Enter on an empty line to send. Type /bye to exit.")

    messages: list[dict] = []

    while True:
        user = read_input()

        # Skip empty submissions instead of sending an empty turn to the
        # model, which wastes a round trip and confuses some models.
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append({"role": "user", "content": user})

        # stream=True returns an iterator of ChatResponse chunks. Each
        # chunk has a small piece of text in chunk.message.content. We
        # print it immediately and also accumulate it so we have the
        # complete reply to append to history when the stream ends.
        full_reply = ""
        for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
            piece = chunk.message.content
            print(piece, end="", flush=True)
            full_reply += piece

        # Print a final newline so the next "> " prompt starts on its
        # own line. Without this the prompt would butt up against the
        # last character of the reply.
        print()

        messages.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    main()
