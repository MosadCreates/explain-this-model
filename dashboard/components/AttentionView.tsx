"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AttentionHeadResult, HeadExplanation } from "@/lib/types";
import ExplanationCard from "./ExplanationCard";

interface AttentionViewProps {
  heads: AttentionHeadResult[];
  explanations?: HeadExplanation[];
}

function patternColor(pattern: string): string {
  switch (pattern) {
    case "diagonal": return "bg-purple-100 text-purple-800";
    case "previous_token": return "bg-amber-100 text-amber-800";
    case "first_token": return "bg-green-100 text-green-800";
    case "content_based": return "bg-rose-100 text-rose-800";
    case "diffuse": return "bg-gray-100 text-gray-800";
    default: return "";
  }
}

export default function AttentionView({ heads, explanations }: AttentionViewProps) {
  const [selected, setSelected] = useState<AttentionHeadResult | null>(null);

  if (!heads || heads.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Attention Heads</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No attention head data available.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Attention Heads</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-1">
            {heads.map((head, idx) => {
              const exp = explanations?.find(
                (e) => e.layer_index === head.layer_index && e.head_index === head.head_index
              );
              return (
                <motion.button
                  key={`${head.layer_index}-${head.head_index}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.03 }}
                  onClick={() => setSelected(selected === head ? null : head)}
                  className={cn(
                    "w-full text-left rounded-lg border p-3 transition-colors hover:bg-accent",
                    selected === head && "ring-2 ring-primary"
                  )}
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-mono text-xs text-muted-foreground w-24">
                      L{head.layer_index} H{head.head_index}
                    </span>
                    <Badge variant="outline" className={cn("text-[10px] px-1.5", patternColor(head.pattern_type))}>
                      {head.pattern_type}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      focus {head.focus_score.toFixed(2)}
                    </span>
                    <span className="w-16 text-right text-[10px] text-muted-foreground">
                      #{head.rank}
                    </span>
                  </div>
                  {exp && (
                    <div className="mt-1.5 text-xs text-muted-foreground line-clamp-1">
                      {exp.hypothesis}
                    </div>
                  )}
                  {head.is_induction_head && (
                    <Badge variant="default" className="mt-1 text-[10px] px-1.5">
                      induction head
                    </Badge>
                  )}
                </motion.button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <AnimatePresence>
        {selected && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  Attention Pattern &mdash; L{selected.layer_index} H{selected.head_index}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AttentionMatrixView matrix={selected.attention_matrix} tokens={selected.top_attended_pairs.map(p => p.key_token)} />

                {selected.top_attended_pairs.length > 0 && (
                  <div>
                    <h4 className="text-xs font-medium text-muted-foreground mb-2">Top Attended Pairs</h4>
                    <div className="space-y-1">
                      {selected.top_attended_pairs.slice(0, 5).map((pair, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="font-mono text-muted-foreground">[{pair.query_position}]</span>
                          <span className="font-medium">&ldquo;{pair.query_token}&rdquo;</span>
                          <span className="text-muted-foreground">&rarr;</span>
                          <span className="font-mono text-muted-foreground">[{pair.key_position}]</span>
                          <span className="font-medium">&ldquo;{pair.key_token}&rdquo;</span>
                          <span className="text-muted-foreground ml-auto">{(pair.weight * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {explanations && (() => {
                  const exp = explanations.find(
                    (e) => e.layer_index === selected.layer_index && e.head_index === selected.head_index
                  );
                  return exp ? (
                    <ExplanationCard
                      label={`L${exp.layer_index} H${exp.head_index}`}
                      hypothesis={exp.hypothesis}
                      confidence={exp.confidence}
                      pattern_type={exp.pattern_type}
                      cached={exp.cached}
                    />
                  ) : null;
                })()}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function AttentionMatrixView({ matrix, tokens }: { matrix: number[][]; tokens: string[] }) {
  const n = matrix.length;
  if (n === 0 || n > 20) {
    return <p className="text-xs text-muted-foreground">Matrix too large to display ({n}x{n})</p>;
  }

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
          <React.Fragment key={`row-${i}`}>
            <div className="text-[8px] text-muted-foreground text-right pr-1 truncate" title={tokens[i] || `[${i}]`}>
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
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
