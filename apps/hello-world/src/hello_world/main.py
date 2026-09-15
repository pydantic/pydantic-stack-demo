"""Send one log line and one span to Logfire.

This is the whole Logfire onboarding: `configure()` once, then `logfire.info()` / `logfire.span()`
anywhere. Everything else in this repo builds on these two calls.
"""

import logfire

from demo_core import configure_logfire


def main() -> None:
    configure_logfire('hello-world', instrument_pydantic_ai=False)

    with logfire.span('greeting {place}', place='world'):
        logfire.info('hello {place}', place='world')

    logfire.force_flush()


if __name__ == '__main__':
    main()
