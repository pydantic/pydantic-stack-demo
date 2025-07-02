import { useState } from 'react';
import Input from './Input';
import Card from './Card';
import { convertTimeInterval } from '../api';
import type { TimeInterval, ConversionError } from '../api';

export default function TimeConverter() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TimeInterval | ConversionError | null>(null);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await convertTimeInterval(input);
      setResult(response);
    } catch {
      setResult({ error: 'An unexpected error occurred' });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInput('');
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-stone-100 p-4">
      <div className="w-full max-w-md mx-auto space-y-6" style={{ paddingTop: '150px' }}>
        <Input
          label="Describe time interval"
          placeholder="e.g., last week, yesterday, this month..."
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          loading={loading}
          showClear={!!result}
          onClear={handleClear}
          autoFocus
        />

        {result && (
          <Card>
            {'error' in result ? (
              <div className="text-red-600">
                <h3 className="font-medium mb-2">Error</h3>
                <p className="text-sm">{result.error}</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Start Time
                    </label>
                    <p className="text-sm text-gray-900 font-mono bg-gray-50 p-2 rounded">
                      {result.startTimestamp}
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      End Time
                    </label>
                    <p className="text-sm text-gray-900 font-mono bg-gray-50 p-2 rounded">
                      {result.endTimestamp}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">Explanation</h3>
                    <p className="text-gray-700">{result.explanation}</p>
                  </div>
                </div>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}
