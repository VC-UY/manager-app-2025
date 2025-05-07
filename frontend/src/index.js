// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { WorkflowProvider } from './context/WorkflowContext';
import './styles/global.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <WorkflowProvider>
      <App />
    </WorkflowProvider>
  </React.StrictMode>
);