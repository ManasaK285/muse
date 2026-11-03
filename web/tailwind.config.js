module.exports = {
  content: ["./web/index.html"],
  theme: {
    extend: {
      fontFamily: { sans: ["IBM Plex Sans", "Inter", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"], mono: ["IBM Plex Mono", "JetBrains Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"] },
      colors: {
        ink: { 900: "#07090c", 800: "#0d1116", 700: "#131820", 600: "#1b222c", 500: "#273140", 400: "#3a4656" },
        txt: { hi: "#e8edf3", mid: "#a3afbf", lo: "#6b788a" },
        ok: "#3fca7d", bad: "#f0524f", warn: "#f2b33d", accent: "#6cb6ff",
      },
    },
  },
};
