# Distributed tracing demo

This project demonstrates distributed Logfire tracing from the browser (React
app) to the backend (FastAPI Python app using PydanticAI).

# Key concepts

The React application (`frontend/`) is instrumented using the
`@pydantic/logfire-browser` NPM package. Details about the instrumentation
specifics can be found in the `ClientInstrumentationProvider.tsx` file.

For security/performance reasons, the browser instrumentation does not send traces to Logfire directly, but
instead sends them to the backend. The backend then proxies the traces to
Logfire - the `/client-traces` FastAPI endpoint in `main.py`. In that way, the Logfire write token is not exposed in the client-side bundle. 
In production, you should ensure that the proxy endpoint is Auth protected and only accessible to your frontend.

# Running the demo

To run the demo, you need to have Python 3.10+ and Node.js 22+ installed. First,
install the dependencies:

```bash
make install
```

Then, export the following environment variables:

```bash
export LOGFIRE_TOKEN=<your-logfire-token>
export LOGFIRE_BASE_URL=https://logfire-us.pydantic.dev/ # e.g. https://logfire-eu.pydantic.dev/ or https://logfire-us.pydantic.dev/
export OPENAI_API_KEY=<your-openai-api-key> # you can get one from https://platform.openai.com/account/api-keys
```

Run the backend:

```bash
make backend-dev
```

Then, in another shell, run the frontend:

```bash
make frontend-dev
```

The App will be available at (http://localhost:5173)[http://localhost:5173]. You
can open your Logfire live view and interact with the App to see the traces in
real-time.

# Further reading

- [Distributed tracing with Logfire](https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/)
- [Logfire Python SDK](https://github.com/pydantic/logfire)
- [Logfire JS](https://github.com/pydantic/logfire-js)
