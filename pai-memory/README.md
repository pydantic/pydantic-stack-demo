# Pydantic AI Memory Examples

Two demos showing different approaches to implementing persistent memory for Pydantic AI agents:

## Message History Approach
### File `with_messages.py`
Stores complete conversation history as JSON in PostgreSQL. The agent receives all past messages as context automatically, making it remember previous interactions naturally.

## Tool-Based Memory Approach
### File `with_tools.py`
Provides the agent with `record_memory` and `retrieve_memories` tools. The agent actively decides what information to store and when to retrieve it from the database.

**Key difference:** Message history provides full context automatically, while tool-based memory gives the agent selective control over what to remember and recall.

## Setup

Run postgres in docker:

```bash
docker run -e POSTGRES_HOST_AUTH_METHOD=trust --rm -it --name pg -p 5432:5432 -d postgres
```
