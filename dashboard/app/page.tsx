"use client";

import { useAnalysisStore } from "@/store/analysis";
import Layout from "@/components/Layout";
import AnalysisForm from "@/components/AnalysisForm";
import ResultsDashboard from "@/components/ResultsDashboard";
import { Button } from "@/components/ui/button";

const SUGGESTED_MODELS = [
  { id: "gpt2", label: "GPT-2" },
  { id: "distilbert-base-uncased", label: "DistilBERT" },
  { id: "facebook/opt-350m", label: "OPT-350M" },
  { id: "google/gemma-2b", label: "Gemma-2B" },
];

export default function Home() {
  const { status, modelName, setModelName } = useAnalysisStore();
  const hasJob = status === "running" || status === "completed" || status === "failed";

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Explain This Model</h1>
          <p className="text-muted-foreground mt-1">
            Upload any HuggingFace transformer model and a prompt &mdash; get a ranked, visualised breakdown of which neurons and attention heads fired most strongly, with natural-language explanations.
          </p>
        </div>

        <AnalysisForm />

        {!modelName && (
          <div className="text-center space-y-2">
            <p className="text-sm text-muted-foreground">Try a popular model:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTED_MODELS.map((m) => (
                <Button
                  key={m.id}
                  variant="outline"
                  size="sm"
                  onClick={() => setModelName(m.id)}
                >
                  {m.label}
                </Button>
              ))}
            </div>
          </div>
        )}

        {hasJob && <ResultsDashboard />}
      </div>
    </Layout>
  );
}
