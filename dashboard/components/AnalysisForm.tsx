"use client";

import { useState } from "react";
import { useAnalysisStore } from "@/store/analysis";
import { submitAnalysis, invalidateJob } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import ModelSearch from "./ModelSearch";
import PromptInput from "./PromptInput";

export default function AnalysisForm() {
  const { modelName, prompt, status, setModelName, setPrompt, setJobId, setStatus, setError, setResult } = useAnalysisStore();
  const [localModel, setLocalModel] = useState(modelName);
  const [localPrompt, setLocalPrompt] = useState(prompt);

  const isRunning = status === "submitting" || status === "running";

  async function handleSubmit() {
    if (!localModel.trim() || !localPrompt.trim()) return;

    setModelName(localModel);
    setPrompt(localPrompt);
    setStatus("submitting");
    setError(null);
    setResult(null);

    try {
      const resp = await submitAnalysis({
        model_name: localModel.trim(),
        prompt: localPrompt.trim(),
      });
      setJobId(resp.job_id);
      setStatus("running");
      invalidateJob(resp.job_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to submit analysis";
      setError(msg);
      setStatus("failed");
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Analyse a Model</CardTitle>
        <CardDescription>
          Enter a HuggingFace model ID and a prompt to see which neurons and attention heads activate most strongly.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Model</label>
          <ModelSearch value={localModel} onChange={setLocalModel} disabled={isRunning} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Prompt</label>
          <PromptInput value={localPrompt} onChange={setLocalPrompt} disabled={isRunning} />
        </div>
        <Button
          onClick={handleSubmit}
          disabled={isRunning || !localModel.trim() || !localPrompt.trim()}
          className="w-full"
        >
          {status === "submitting" ? "Submitting..." : status === "running" ? "Running..." : "Analyse"}
        </Button>
      </CardContent>
    </Card>
  );
}
