"""
Entry point for La Vie En Rose adjoint method comparison server.

Usage:
    python main.py [--host HOST] [--port PORT]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="La Vie En Rose — Adjoint Method Comparison Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
