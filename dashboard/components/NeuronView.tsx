"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartTooltip,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { NeuronResult, NeuronExplanation } from "@/lib/types";
import ExplanationCard from "./ExplanationCard";

interface NeuronViewProps {
  neurons: NeuronResult[];
  explanations?: NeuronExplanation[];
}

function activationColor(val: number, max: number): string {
  const intensity = max > 0 ? val / max : 0;
  if (intensity < 0.3) return "bg-blue-100 text-blue-800";
  if (intensity < 0.6) return "bg-blue-200 text-blue-800";
  if (intensity < 0.8) return "bg-blue-400 text-white";
  return "bg-blue-600 text-white";
}

export default function NeuronView({ neurons, explanations }: NeuronViewProps) {
  const [selected, setSelected] = useState<NeuronResult | null>(null);

  if (!neurons || neurons.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Neuron Activations</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No neuron data available.</p>
        </CardContent>
      </Card>
    );
  }

  const maxAct = Math.max(...neurons.map((n) => n.max_activation), 1);

  const heatmapData = selected
    ? selected.activation_values_per_token.map((val, i) => ({
        position: i,
        activation: val,
        token: selected.context_window[i] || `[pos ${i}]`,
      }))
    : [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Neuron Activations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-1">
            {neurons.map((neuron, idx) => {
              const exp = explanations?.find(
                (e) => e.layer_index === neuron.layer_index && e.neuron_index === neuron.neuron_index
              );
              return (
                <motion.button
                  key={`${neuron.layer_index}-${neuron.neuron_index}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.03 }}
                  onClick={() => setSelected(selected === neuron ? null : neuron)}
                  className={cn(
                    "w-full text-left rounded-lg border p-3 transition-colors hover:bg-accent",
                    selected === neuron && "ring-2 ring-primary"
                  )}
                >
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-mono text-xs text-muted-foreground w-24">
                      L{neuron.layer_index} N{neuron.neuron_index}
                    </span>
                    <span className="flex-1 truncate px-2 text-xs">
                      &ldquo;{neuron.activating_token}&rdquo;
                    </span>
                    <Badge
                      variant="secondary"
                      className={cn(
                        "text-[10px] px-1.5",
                        activationColor(neuron.max_activation, maxAct)
                      )}
                    >
                      {neuron.max_activation.toFixed(2)}
                    </Badge>
                    <span className="w-16 text-right text-[10px] text-muted-foreground">
                      #{neuron.rank}
                    </span>
                  </div>
                  {exp && (
                    <div className="mt-1.5 text-xs text-muted-foreground line-clamp-1">
                      {exp.hypothesis}
                    </div>
                  )}
                </motion.button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <AnimatePresence>
        {selected && heatmapData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  Per-Token Activation &mdash; L{selected.layer_index} N{selected.neuron_index}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={heatmapData}>
                    <XAxis
                      dataKey="position"
                      tick={{ fontSize: 10 }}
                      tickFormatter={(v) => selected.context_window[v] || `[${v}]`}
                    />
                    <YAxis tick={{ fontSize: 10 }} />
                    <RechartTooltip
                      formatter={(value: number) => [value.toFixed(4), "Activation"]}
                      labelFormatter={(label) =>
                        `${selected.context_window[label as number] || `[${label}]`} (pos ${label})`
                      }
                    />
                    <Bar dataKey="activation" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>

                {explanations && (
                  <div className="mt-4">
                    {(() => {
                      const exp = explanations.find(
                        (e) =>
                          e.layer_index === selected.layer_index &&
                          e.neuron_index === selected.neuron_index
                      );
                      return exp ? (
                        <ExplanationCard
                          label={`L${exp.layer_index} N${exp.neuron_index}`}
                          hypothesis={exp.hypothesis}
                          confidence={exp.confidence}
                          pattern_type={exp.pattern_type}
                          cached={exp.cached}
                        />
                      ) : null;
                    })()}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
