"""Simple terminal entry point."""

from .service import BanoQabilServiceDesk


def main() -> None:
    service = BanoQabilServiceDesk()
    print("Bano Qabil AI Service Desk — Gemini Python implementation")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            print(f"\nAI: {service.answer(question)}")
        except Exception as exc:
            print(f"\nAI: The service could not complete the request: {exc}")


if __name__ == "__main__":
    main()
