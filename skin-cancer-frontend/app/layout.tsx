import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Skin Cancer Detection",
  description: "AI-based skin disease prediction frontend"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
