import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Acoustic Anomaly Detection",
  description: "Modern UI starter for machine sound anomaly detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
