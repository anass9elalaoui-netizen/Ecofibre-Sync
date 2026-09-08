/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'eco-dark': '#0f172a',
        'eco-primary': '#10b981',
        'eco-secondary': '#3b82f6',
      }
    },
  },
  plugins: [],
}
