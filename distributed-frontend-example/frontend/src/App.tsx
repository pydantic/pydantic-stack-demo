import { useForm } from '@tanstack/react-form'
import { useState } from 'react';


function App() {
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const form = useForm({
    defaultValues: {
      imagePrompt: '',
    },
    onSubmit: async ({ value }) => {
      const params = new URLSearchParams({ prompt: value.imagePrompt });

      const response = await fetch(`${import.meta.env.VITE_FAST_API_BACKEND_BASE_URL}/generate?${params.toString()}`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const { nextUrl } = await response.json();
      setGeneratedImageUrl(`${import.meta.env.VITE_FAST_API_BACKEND_BASE_URL}${nextUrl}`);
    },
  })


  return (
    <div className="container">
      <h1>Image Generator</h1>

      {
        generatedImageUrl ? (
          <div id="imageSection" className="image-container">
            <img
              id="generatedImage"
              src={generatedImageUrl}
              className="generated-image"
              alt="Generated image"
            />
            <br /><br />
            <button role='button' type='button' className="clear-btn" onClick={() => setGeneratedImageUrl(null)}>Clear</button>
          </div>)

          :
          (
            <div id="promptSection" className="prompt-section">
              <form onSubmit={(e) => {
                e.preventDefault()
                e.stopPropagation()
                form.handleSubmit()
              }}>
                <form.Field name="imagePrompt" validators={{
                  onChange: ({ value }) => {
                    return value.trim() !== '' ? undefined : 'Please enter a prompt'
                  }
                }}>
                  {(field) => {
                    return <><input
                      type="text"
                      id={field.name}
                      value={field.state.value}
                      onBlur={field.handleBlur}
                      onChange={(e) => field.handleChange(e.target.value)}
                      placeholder="Painting of an iphone in the style of Titian..."
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
                    <button id="generateBtn" type="submit" aria-disabled={!canSubmit} disabled={!canSubmit || isSubmitting}>
                      {isSubmitting ? 'Generating...' : 'Generate Image'}
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
