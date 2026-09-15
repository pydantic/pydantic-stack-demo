import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ClientInstrumentationProvider from './ClientInstrumentationProvider.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ClientInstrumentationProvider>
      <App />
    </ClientInstrumentationProvider>
  </StrictMode>,
)
