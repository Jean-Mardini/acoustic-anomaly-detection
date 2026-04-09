export default function Home() {
  return (
    <main className="min-h-screen p-8 md:p-12">
      <div className="mx-auto max-w-5xl space-y-6">
        <h1 className="text-3xl font-semibold md:text-5xl">
          First-Shot Acoustic Anomaly Detection
        </h1>
        <p className="text-zinc-300">
          Step 0 UI starter: modern foundation ready for waveform, spectrogram,
          anomaly gauge, and explanation visuals.
        </p>

        <section className="grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-zinc-700 bg-card p-4">
            Audio Upload (placeholder)
          </div>
          <div className="rounded-xl border border-zinc-700 bg-card p-4">
            Waveform & Spectrogram (placeholder)
          </div>
          <div className="rounded-xl border border-zinc-700 bg-card p-4">
            Decision & Insights (placeholder)
          </div>
        </section>
      </div>
    </main>
  );
}
