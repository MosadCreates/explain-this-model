export interface AnalyzeRequest {
  model_name: string;
  prompt: string;
}

export interface AnalyzeResponse {
  job_id: string;
  status: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  model_name: string;
  created_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  result_url: string | null;
}

export interface ResultResponse {
  job_id: string;
  status: string;
  result: AnalysisResult | null;
  error_message: string | null;
}

export interface AnalysisResult {
  model_name: string;
  prompt: string;
  tokens: string[];
  architecture_type: string;
  parameter_count: number;
  n_layers: number;
  n_heads: number;
  neuron_results: NeuronResult[];
  attention_results: AttentionHeadResult[];
  layer_summaries: LayerSummary[];
  neuron_count: number;
  head_count: number;
  top_neuron_explanation: string;
  total_dead_neurons: number;
  analysis_duration_seconds: number;
  explanations: Explanations;
}

export interface NeuronResult {
  layer_index: number;
  neuron_index: number;
  max_activation: number;
  mean_activation: number;
  std_activation: number;
  fraction_active: number;
  activating_token: string;
  activating_token_position: number;
  context_window: string[];
  context_window_positions: number[];
  activation_values_per_token: number[];
  is_dead: boolean;
  z_score: number;
  rank: number;
}

export interface AttentionHeadResult {
  layer_index: number;
  head_index: number;
  focus_score: number;
  entropy: number;
  pattern_type: string;
  attention_matrix: number[][];
  top_attended_pairs: AttendedPair[];
  is_induction_head: boolean;
  max_attention_weight: number;
  attending_entropy: number;
  rank: number;
}

export interface AttendedPair {
  query_position: number;
  key_position: number;
  query_token: string;
  key_token: string;
  weight: number;
}

export interface LayerSummary {
  layer_index: number;
  total_neurons: number;
  dead_neurons: number;
  max_activation: number;
  mean_activation: number;
  fraction_dead: number;
}

export interface Explanations {
  neurons: NeuronExplanation[];
  heads: HeadExplanation[];
  total_api_calls: number;
  total_cached: number;
  explanation_duration_seconds: number;
}

export interface NeuronExplanation {
  layer_index: number;
  neuron_index: number;
  hypothesis: string;
  confidence: string;
  pattern_type: string;
  cached: boolean;
}

export interface HeadExplanation {
  layer_index: number;
  head_index: number;
  hypothesis: string;
  confidence: string;
  pattern_type: string;
  cached: boolean;
}

export interface ModelSearchResult {
  model_id: string;
  architecture: string | null;
  likes: number | null;
}

export interface HealthResponse {
  status: string;
  version: string;
}
