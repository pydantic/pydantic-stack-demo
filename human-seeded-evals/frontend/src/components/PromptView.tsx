import React, { useState, useEffect } from 'react';
import { getFields, submitContext, updateContext, type Field } from '../api';

export function PromptView() {
  const [fields, setFields] = useState<Field[]>([]);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [improving, setImproving] = useState(false);

  const loadFields = async () => {
    setLoading(true);
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
  };

  useEffect(() => {
    document.title = 'Agent Context Form';
    loadFields();
  }, []);

  const handleInputChange = (fieldId: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [fieldId]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      await submitContext(formData);
      console.log('Form submitted successfully');
      // Reload data after successful submission
      await loadFields();
    } catch (error) {
      console.error('Error submitting form:', error);
      // Handle error (e.g., show error message)
    } finally {
      setSubmitting(false);
    }
  };

  const handleImproveContext = async () => {
    setImproving(true);

    try {
      await updateContext();
      console.log('Context updated successfully');
      // Reload data after successful update
      await loadFields();
    } catch (error) {
      console.error('Error updating context:', error);
      // Handle error (e.g., show error message)
    } finally {
      setImproving(false);
    }
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
          <button
            onClick={handleImproveContext}
            disabled={improving || loading}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg font-medium transition-colors hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {improving ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Improving...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Improve Agent Context</span>
              </>
            )}
          </button>
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
                    rows={4}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-vertical"
                  />
                </div>
              ))}

              <div className="pt-4">
                <button
                  type="submit"
                  disabled={submitting}
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-medium transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed"
                >
                  {submitting ? (
                    <div className="flex items-center justify-center">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Submitting...
                    </div>
                  ) : (
                    'Submit'
                  )}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
