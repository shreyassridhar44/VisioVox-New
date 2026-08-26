import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import './globals.css';

export const metadata: Metadata = {
  title: 'VisioVox',
  description: 'Hear one speaker at a time.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="header">
          <a href="/" className="brand">
            VisioVox
          </a>
          <nav>
            <a href="/projects">Projects</a>
          </nav>
        </header>
        <main className="main">{children}</main>
      </body>
    </html>
  );
}
