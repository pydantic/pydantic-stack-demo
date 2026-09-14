import { getWebAutoInstrumentations } from "@opentelemetry/auto-instrumentations-web";
import * as logfire from '@pydantic/logfire-browser';
import { type ReactNode, useEffect, useRef } from "react";
import { BACKEND_BASE_URL } from "./config";


interface ClientInstrumentationProviderProps {
  children: ReactNode;
}

export default function ClientInstrumentationProvider({ children }: ClientInstrumentationProviderProps) {
  const logfireConfigured = useRef<boolean>(false);

  useEffect(() => {
    if (!logfireConfigured.current) {
      // Send browser spans to our FastAPI proxy, which adds the Logfire write token server-side.
      const url = new URL('/client-traces', BACKEND_BASE_URL);
      logfire.configure({
        traceUrl: url.toString(),
        serviceName: 'frontend',
        serviceVersion: '0.1.0',
        // For the demo we export each span immediately; batch in production.
        batchSpanProcessorConfig: {
          maxExportBatchSize: 1,
        },
        instrumentations: [
          getWebAutoInstrumentations({
            "@opentelemetry/instrumentation-fetch": {
              // Attach `traceparent` to fetches to any origin (the backend is cross-origin in dev).
              propagateTraceHeaderCorsUrls: /.*/
            },
            // Clicks and page-load timings are noise for this demo.
            "@opentelemetry/instrumentation-user-interaction": {
              enabled: false
            },
            "@opentelemetry/instrumentation-document-load": {
              enabled: false
            }
          })
        ],
        diagLogLevel: logfire.DiagLogLevel.ALL
      })
      logfireConfigured.current = true;
    }
  }, []);

  return children;
}
