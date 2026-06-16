"use client";

import Link from "next/link";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <Link href="/" className="font-bold text-lg tracking-tight">
            Explain This Model
          </Link>
          <span className="text-xs text-muted-foreground">Interpretability-as-a-Service</span>
        </div>
      </header>
      <main className="flex-1 container mx-auto px-4 py-6">{children}</main>
      <footer className="border-t py-4 text-center text-xs text-muted-foreground space-y-1">
        <p>Analysis runs on CPU. No data is stored permanently.</p>
        <p className="text-amber-600 dark:text-amber-400 font-medium">
          Disclaimer: Natural-language explanations are AI-generated hypotheses, not ground truth. Interpretability is an active research area.
        </p>
      </footer>
    </div>
  );
}
