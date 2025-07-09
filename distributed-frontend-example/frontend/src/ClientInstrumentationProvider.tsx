import { getWebAutoInstrumentations } from "@opentelemetry/auto-instrumentations-web";
import * as logfire from '@pydantic/logfire-browser';
import { type ReactNode, useEffect, useRef } from "react";


interface ClientInstrumentationProviderProps {
  children: ReactNode;
}

export default function ClientInstrumentationProvider({ children }: ClientInstrumentationProviderProps) {
  const logfireConfigured = useRef<boolean>(false);

  useEffect(() => {
    if (!logfireConfigured.current) {
      // use our FastAPI proxy to send traces to Logfire
      const url = new URL('/client-traces', import.meta.env.VITE_FAST_API_BACKEND_BASE_URL);
      logfire.configure({
        traceUrl: url.toString(),
        serviceName: 'image-generator-frontend',
        serviceVersion: '0.1.0',
        // for development purposes, we want to see traces as soon as they happen, 
        // in production, we want to batch traces and send them in batches
        batchSpanProcessorConfig: {
          maxExportBatchSize: 1,
          scheduledDelayMillis: 50,
        },
        instrumentations: [
          getWebAutoInstrumentations({
            "@opentelemetry/instrumentation-fetch": {
              propagateTraceHeaderCorsUrls: /.*/
            }, 
            // disable user interaction instrumentation, clicks are not relevant for us
            "@opentelemetry/instrumentation-user-interaction": {
              enabled: false
            }, 
            // useful in general, disabling it for the demo purposes
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
