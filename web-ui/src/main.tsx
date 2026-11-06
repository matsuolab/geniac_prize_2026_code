import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { adoptTokenFromUrl } from './api'
import './index.css'

const root = ReactDOM.createRoot(document.getElementById('root')!)
let generation = 0

function render(): void {
  root.render(
    <React.StrictMode>
      <App key={generation++} />
    </React.StrictMode>,
  )
}

adoptTokenFromUrl()
render()
// Opening the printed URL in a tab already on this page only changes the fragment.
// Start the app again in place with the new token; a reload would lose a token that
// could only be kept in memory.
window.addEventListener('hashchange', () => {
  if (adoptTokenFromUrl()) render()
})
