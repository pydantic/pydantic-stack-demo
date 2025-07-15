import React, { useState, useEffect } from 'react';
import { getFields, type Field } from '../api';

export function PromptView() {
  const [fields, setFields] = useState<Field[]>([]);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadFields() {
      try {
        const fieldsData = await getFields();
        setFields(fieldsData);
        const initialData: Record<string, string> = {};
        fieldsData.forEach(field => {
          initialData[field.id] = field.text;
        });
        setFormData(initialData);
      } finally {
        setLoading(false);
      }
    }
    loadFields();
  }, []);

  const handleInputChange = (fieldId: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [fieldId]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
    // Handle form submission here
  };

  const handleBack = () => {
    window.location.href = '/';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-lg text-white">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8 bg-gray-900">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={handleBack}
            className="flex items-center space-x-2 px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <span>Back</span>
          </button>
          <h1 className="text-3xl font-bold text-white">Agent Context Form</h1>
          <div className="w-20"></div>
        </div>

        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          {fields.length === 0 ? (
            <div className="text-center text-gray-400 text-lg">
              No agent context available
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              {fields.map(field => (
                <div key={field.id}>
                  <label
                    htmlFor={field.id}
                    className="block text-sm font-medium mb-2 text-gray-200 whitespace-pre-wrap break-words"
                  >
                    {field.id}
                  </label>
                  <textarea
                    id={field.id}
                    value={formData[field.id] || ''}
                    onChange={(e) => handleInputChange(field.id, e.target.value)}
                    placeholder={`Enter ${field.id.toLowerCase()}`}
                    rows={4}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-vertical"
                  />
                </div>
              ))}

              <div className="pt-4">
                <button
                  type="submit"
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-medium transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  Submit
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
