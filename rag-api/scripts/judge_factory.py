"""Project LLM factory for opt-in semantic evaluation judging."""


def create_llm():
    """Return the application's configured LLM without duplicating credentials."""
    from app import app_manager

    return app_manager.llm
