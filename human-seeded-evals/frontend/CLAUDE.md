# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Scripts
- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production (runs TypeScript check + Vite build)
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint on all TypeScript/TSX files
- `npm run typecheck` - Run TypeScript compiler for type checking only

### Code Quality
Always run both `npm run lint` and `npm run typecheck` before committing changes to ensure code quality.

## Technology Stack

**Frontend Framework:** React 19 with TypeScript
**Build Tool:** Vite 7
**Styling:** Tailwind CSS 4
**Type Checking:** TypeScript 5.8
**Code Quality:** ESLint 9 with TypeScript ESLint, React Hooks, and React Refresh plugins

## Project Structure

```
src/
├── main.tsx      # Application entry point
├── App.tsx       # Root React component
├── main.css      # Global styles (Tailwind imports)
└── vite-env.d.ts # Vite type definitions
```

## Configuration Files

- **vite.config.js** - Vite configuration with Tailwind CSS and React plugins
- **tsconfig.json** - TypeScript configuration with strict type checking
- **eslint.config.js** - ESLint configuration using TypeScript ESLint
- **package.json** - Dependencies and npm scripts

## Development Notes

- Uses React 19's latest features
- Strict TypeScript configuration with `noUnusedLocals` and `noUnusedParameters`
- ESLint configured for TypeScript files (`.ts`, `.tsx`) only
- Tailwind CSS 4 integrated via Vite plugin
- No testing framework currently configured

You are in a minimal typescript react app with tailwindcss.

Use dummy data while building this app, put all the dummy data in a single file `api.ts` with async methods to "fetch" data so it's easy to replace with real API calls later.

Run `npm run typecheck` and `npm run lint` regularly to check for errors.

Before building pages, define the set of common components that will be used across the app, such as buttons, input fields, and cards. Then use these whenever possible to avoid duplication.

Use tailwindcss for styling.

The app should be clean and modern with no drop shadows or gradients, the background color should be a light cream color, and the main content, should have white backgrounds. The text should be the "IMB Plex Sans" font from google fonts.
