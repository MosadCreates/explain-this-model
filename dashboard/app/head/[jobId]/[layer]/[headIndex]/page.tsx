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
import ExplanationCard from "@/components/ExplanationCard";

function AttentionMatrixView({ matrix, tokens }: { matrix: number[][]; tokens: string[] }) {
  const n = matrix.length;
  if (n === 0) return <p className="text-xs text-muted-foreground">Empty matrix</p>;
  const maxVal = Math.max(...matrix.flat(), 0.01);

  return (
    <div className="overflow-x-auto">
      <div className="inline-grid" style={{ gridTemplateColumns: `24px repeat(${n}, minmax(18px,1fr))` }}>
        <div />
        {Array.from({ length: n }).map((_, i) => (
          <div key={i} className="text-[8px] text-muted-foreground text-center truncate" title={tokens[i] || `[${i}]`}>
            {tokens[i] || `[${i}]`}
          </div>
        ))}
        {matrix.map((row, i) => (
          <>
            <div key={`label-${i}`} className="text-[8px] text-muted-foreground text-right pr-1 truncate" title={tokens[i] || `[${i}]`}>
              {tokens[i] || `[${i}]`}
            </div>
            {row.map((cell, j) => (
              <div
                key={`${i}-${j}`}
                className="aspect-square rounded-sm"
                style={{
                  backgroundColor: `rgba(59, 130, 246, ${Math.min(cell / maxVal, 1)})`,
                  opacity: cell > 0.01 ? 0.9 : 0.15,
                }}
                title={`[${i}]->[${j}]: ${(cell * 100).toFixed(1)}%`}
              />
            ))}
          </>
        ))}
      </div>
    </div>
  );
}

export default function HeadDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const layerIdx = parseInt(params.layer as string);
  const headIdx = parseInt(params.headIndex as string);

  const { data, error, isLoading } = useResult(jobId);

  const head = useMemo(() => {
    if (!data?.result) return null;
    return data.result.attention_results.find(
      (h) => h.layer_index === layerIdx && h.head_index === headIdx
    ) || null;
  }, [data, layerIdx, headIdx]);

  const explanation = useMemo(() => {
    if (!data?.result?.explanations?.heads) return null;
    return data.result.explanations.heads.find(
      (e) => e.layer_index === layerIdx && e.head_index === headIdx
    ) || null;
  }, [data, layerIdx, headIdx]);

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

  if (error || !head) {
    return (
      <Layout>
        <div className="max-w-3xl mx-auto text-center py-12">
          <p className="text-destructive">Attention head not found or error loading results.</p>
          <Link href="/">
            <Button variant="outline" className="mt-4">Back to Home</Button>
          </Link>
        </div>
      </Layout>
    );
  }

  const tokens = data?.result?.tokens || [];
  const tokenLabels = tokens.length > 0 ? tokens : head.top_attended_pairs.map(p => p.key_token);

  return (
    <Layout>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">&larr; Back</Link>
            <h1 className="text-2xl font-bold mt-1">
              Head L{head.layer_index} H{head.head_index}
            </h1>
            <p className="text-sm text-muted-foreground">{data?.result?.model_name}</p>
          </div>
          <Badge variant={head.pattern_type === "content_based" ? "default" : "secondary"} className="text-sm">
            {head.pattern_type}
          </Badge>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{(head.focus_score * 100).toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">Focus Score</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{head.entropy.toFixed(3)}</div>
              <div className="text-xs text-muted-foreground">Entropy</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{(head.max_attention_weight * 100).toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">Max Weight</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 text-center">
              <div className="text-lg font-bold">{head.is_induction_head ? "Yes" : "No"}</div>
              <div className="text-xs text-muted-foreground">Induction Head</div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Attention Pattern</CardTitle>
          </CardHeader>
          <CardContent>
            <AttentionMatrixView matrix={head.attention_matrix} tokens={tokenLabels} />
          </CardContent>
        </Card>

        {head.top_attended_pairs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Top Attended Pairs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {head.top_attended_pairs.slice(0, 10).map((pair, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs py-0.5">
                    <span className="font-mono text-muted-foreground w-8">[{pair.query_position}]</span>
                    <span className="font-medium">&ldquo;{pair.query_token}&rdquo;</span>
                    <span className="text-muted-foreground">&rarr;</span>
                    <span className="font-mono text-muted-foreground w-8">[{pair.key_position}]</span>
                    <span className="font-medium">&ldquo;{pair.key_token}&rdquo;</span>
                    <span className="text-muted-foreground ml-auto">{(pair.weight * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {explanation && (
          <ExplanationCard
            label={`L${explanation.layer_index} H${explanation.head_index}`}
            hypothesis={explanation.hypothesis}
            confidence={explanation.confidence}
            pattern_type={explanation.pattern_type}
            cached={explanation.cached}
          />
        )}
      </div>
    </Layout>
  );
}
