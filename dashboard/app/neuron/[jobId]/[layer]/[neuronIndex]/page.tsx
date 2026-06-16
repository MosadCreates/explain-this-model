"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useResult } from "@/lib/api";
import Layout from "@/components/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import ExplanationCard from "@/components/ExplanationCard";

export default function NeuronDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const layerIdx = parseInt(params.layer as string);
  const neuronIdx = parseInt(params.neuronIndex as string);

  const { data, error, isLoading } = useResult(jobId);

  const neuron = useMemo(() => {
    if (!data?.result) return null;
    return data.result.neuron_results.find(
      (n) => n.layer_index === layerIdx && n.neuron_index === neuronIdx
    ) || null;
  }, [data, layerIdx, neuronIdx]);

  const explanation = useMemo(() => {
    if (!data?.result?.explanations?.neurons) return null;
    return data.result.explanations.neurons.find(
      (e) => e.layer_index === layerIdx && e.neuron_index === neuronIdx
    ) || null;
  }, [data, layerIdx, neuronIdx]);

  if (isLoading) {
    return (
      <Layout>
        <div className="max-w-3xl mx-auto space-y-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-64 w-full" />
        </div>
      </Layout>
    );
  }

  if (error || !neuron) {
    return (
      <Layout>
        <div className="max-w-3xl mx-auto text-center py-12">
          <p className="text-destructive">Neuron not found or error loading results.</p>
          <Link href="/">
            <Button variant="outline" className="mt-4">Back to Home</Button>
          </Link>
        </div>
      </Layout>
    );
  }

  const heatmapData = neuron.activation_values_per_token.map((val, i) => ({
    position: i,
    activation: val,
    token: neuron.context_window[i] || `[pos ${i}]`,
  }));

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">&larr; Back</Link>
            <h1 className="text-2xl font-bold mt-1">
              Neuron L{neuron.layer_index} N{neuron.neuron_index}
            </h1>
            <p className="text-sm text-muted-foreground">{data?.result?.model_name}</p>
          </div>
          <Badge variant="secondary" className="text-lg px-3 py-1">
            {neuron.max_activation.toFixed(3)}
          </Badge>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Per-Token Activation Heatmap</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={heatmapData}>
                <XAxis dataKey="position" tick={{ fontSize: 10 }} tickFormatter={(v) => neuron.context_window[v] || `[${v}]`} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), "Activation"]}
                  labelFormatter={(label) => `${neuron.context_window[label as number] || `[${label}]`} (pos ${label})`}
                />
                <Bar dataKey="activation" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{neuron.max_activation.toFixed(3)}</div>
              <div className="text-xs text-muted-foreground">Max Activation</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{neuron.mean_activation.toFixed(4)}</div>
              <div className="text-xs text-muted-foreground">Mean Activation</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{(neuron.fraction_active * 100).toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">Fraction Active</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{neuron.z_score.toFixed(2)}</div>
              <div className="text-xs text-muted-foreground">Z-Score</div>
            </CardContent>
          </Card>
        </div>

        {explanation && (
          <ExplanationCard
            label={`L${explanation.layer_index} N${explanation.neuron_index}`}
            hypothesis={explanation.hypothesis}
            confidence={explanation.confidence}
            pattern_type={explanation.pattern_type}
            cached={explanation.cached}
          />
        )}

        <div className="flex gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">Active tokens: {neuron.context_window.length}</Badge>
          <Badge variant="outline">Peak token: &ldquo;{neuron.activating_token}&rdquo;</Badge>
          <Badge variant="outline">Rank #{neuron.rank}</Badge>
        </div>
      </div>
    </Layout>
  );
}
