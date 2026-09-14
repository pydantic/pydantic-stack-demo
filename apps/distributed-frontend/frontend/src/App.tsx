import { useForm } from '@tanstack/react-form'
import { useState } from 'react';
import { BACKEND_BASE_URL } from './config';


function App() {
  const [answer, setAnswer] = useState<string | null>(null);
  const form = useForm({
    defaultValues: {
      question: '',
    },
    onSubmit: async ({ value }) => {
      // This fetch is instrumented by @opentelemetry/instrumentation-fetch, so it carries a
      // `traceparent` header and the backend's spans join the browser's trace.
      const response = await fetch(`${BACKEND_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: value.question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const { answer } = await response.json();
      setAnswer(answer);
    },
  })


  return (
    <div className="container">
      <h1>Ask the backend</h1>

      {
        answer ? (
          <div id="answerSection" className="answer-container">
            <p id="answer" className="answer">{answer}</p>
            <button role='button' type='button' className="clear-btn" onClick={() => setAnswer(null)}>Ask another</button>
          </div>)
          :
          (
            <div id="promptSection" className="prompt-section">
              <form onSubmit={(e) => {
                e.preventDefault()
                e.stopPropagation()
                form.handleSubmit()
              }}>
                <form.Field name="question" validators={{
                  onChange: ({ value }) => {
                    return value.trim() !== '' ? undefined : 'Please enter a question'
                  }
                }}>
                  {(field) => {
                    return <><input
                      type="text"
                      id={field.name}
                      value={field.state.value}
                      onBlur={field.handleBlur}
                      onChange={(e) => field.handleChange(e.target.value)}
                      placeholder="Why is the sky blue?"
                      maxLength={500}
                    />
                      {!field.state.meta.isValid && (
                        <em role="alert">{field.state.meta.errors.join(', ')}</em>
                      )}
                    </>
                  }
                  }
                </form.Field>
                <br />
                <form.Subscribe
                  selector={(state) => [state.canSubmit, state.isSubmitting]}
                  children={([canSubmit, isSubmitting]) => (
                    <button id="askBtn" type="submit" aria-disabled={!canSubmit} disabled={!canSubmit || isSubmitting}>
                      {isSubmitting ? 'Thinking...' : 'Ask'}
                    </button>
                  )}
                />
              </form>
            </div>
          )}
    </div >

  )
}

export default App
