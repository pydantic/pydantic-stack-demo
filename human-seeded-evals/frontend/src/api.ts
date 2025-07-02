export interface TimeInterval {
  startTimestamp: string;
  endTimestamp: string;
  explanation: string;
}

export interface ConversionError {
  error: string;
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
