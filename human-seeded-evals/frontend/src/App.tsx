import { useState, useEffect } from 'react';
import TimeConverter from './components/TimeConverter';
import { PromptView } from './components/PromptView';

function App() {
  const [currentView, setCurrentView] = useState<string>('');

  useEffect(() => {
    const handleHashChange = () => {
      setCurrentView(window.location.hash.slice(1));
    };

    // Set initial view
    handleHashChange();

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange);

    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  if (currentView === 'agent-context') {
    return <PromptView />;
  } else {
    return <TimeConverter />;
  }

}

export default App
