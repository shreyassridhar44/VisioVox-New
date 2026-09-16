/** Tailwind v4 is a PostCSS plugin; its scale is configured in CSS via @theme,
 *  so there is no tailwind.config.js to drift from the token file. */
const config = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};

export default config;
