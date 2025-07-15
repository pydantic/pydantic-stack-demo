export interface TimeInterval {
  startTimestamp: string;
  endTimestamp: string;
  explanation: string;
}

export interface ConversionError {
  error: string;
}

export interface Field {
  id: string; // The label text
  text: string; // Default input value
}

export async function convertTimeInterval(prompt: string): Promise<TimeInterval | ConversionError> {
  try {
    const response = await fetch('/api/timerange', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        error: errorData.error || `Server error: ${response.status}`
      };
    }

    const data = await response.json();
    return data;
  } catch {
    return {
      error: 'Network error: Unable to connect to server'
    };
  }
}

export async function getFields(): Promise<Field[]> {
  const response = await fetch('/api/context');

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server error: ${response.status}`);
  }
  return await response.json();
}

export async function submitContext(formData: Record<string, string>): Promise<void> {
  const fields: Field[] = Object.entries(formData).map(([id, text]) => ({ id, text }));
  const response = await fetch('/api/context', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ fields }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server error: ${response.status}`);
  }
}

export async function updateContext(): Promise<void> {
  const response = await fetch('/api/context/update', {
    method: 'POST'
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server error: ${response.status}`);
  }
}
