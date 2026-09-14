// Where the FastAPI backend runs. Override with VITE_BACKEND_BASE_URL in a frontend/.env.local file.
export const BACKEND_BASE_URL: string = import.meta.env.VITE_BACKEND_BASE_URL ?? 'http://127.0.0.1:8000';
