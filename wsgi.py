"""WSGI entry point used by gunicorn/waitress and the container."""

from research import create_app

app = create_app()


if __name__ == "__main__":
    from research.web import main

    main()
