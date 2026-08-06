<script setup lang="ts">
/**
 * Vector workshop: extraction and training merged into one flow.
 * Pick a method, configure, submit the job to the retained Flask backend,
 * poll status, then hand the produced vector to the playground.
 */
import { computed, onBeforeUnmount, ref } from "vue";
import { useRouter } from "vue-router";
import StringListEditor from "../components/StringListEditor.vue";
import { builtinExtractionPresets, builtinTrainingPresets } from "../data/builtinConfigs";
import { useI18n } from "../i18n";
import * as flask from "../lib/flask";
import { loadCustomSpec } from "../lib/playgroundStore";
import { defaultApplySpec, defaultSteeringSpec } from "../lib/spec";

const router = useRouter();
const { t } = useI18n();

type JobKind = "extraction" | "training";
const kind = ref<JobKind>("extraction");

// ---- Extraction form state ----
const extraction = ref({
  model_path: "",
  gpu_devices: "0",
  method: "diffmean" as "diffmean" | "pca" | "lat",
  token_pos: -1,
  normalize: true,
  positive_samples: [""] as string[],
  negative_samples: [""] as string[],
  output_path: "results/my_vector.gguf",
});

// ---- Training form state ----
const training = ref({
  model_path: "",
  gpu_devices: "0",
  intervention: "loreft",
  layer: 8,
  component: "block_output",
  low_rank_dimension: 4,
  num_train_epochs: 100,
  per_device_train_batch_size: 10,
  learning_rate: 0.004,
  logging_steps: 40,
  output_dir: "results/my_training",
  examples: [["", ""]] as [string, string][],
});

// ---- Presets ----
// Built-in presets ship with the app; the job backend may serve more.
const serverPresets = ref<{ name: string; display_name?: string }[]>([]);
const selectedPreset = ref("");
const presetError = ref("");
const presetsUnavailable = ref(false);

const builtinPresets = computed(() =>
  kind.value === "extraction" ? builtinExtractionPresets : builtinTrainingPresets,
);

/** Backend presets that aren't already shipped (same files, same names). */
const serverOnlyPresets = computed(() =>
  serverPresets.value.filter((p) => !builtinPresets.value.some((b) => b.name === p.name)),
);

async function refreshPresets(): Promise<void> {
  presetError.value = "";
  presetsUnavailable.value = false;
  selectedPreset.value = "";
  try {
    const resp =
      kind.value === "extraction"
        ? await flask.listExtractionConfigs()
        : await flask.listTrainingConfigs();
    serverPresets.value = resp.configs;
  } catch {
    // An unreachable job backend is a normal state for a static review
    // deployment; the built-in presets stay usable either way.
    serverPresets.value = [];
    presetsUnavailable.value = true;
  }
}

async function importPreset(): Promise<void> {
  const selected = selectedPreset.value;
  if (!selected) return;
  presetError.value = "";
  const separator = selected.indexOf(":");
  const scope = selected.slice(0, separator);
  const name = selected.slice(separator + 1);
  try {
    if (kind.value === "extraction") {
      const cfg =
        scope === "builtin"
          ? builtinExtractionPresets.find((p) => p.name === name)?.config
          : await flask.getExtractionConfig(name);
      if (!cfg) throw new Error(`unknown preset ${name}`);
      extraction.value = {
        model_path: cfg.model_path ?? "",
        gpu_devices: cfg.gpu_devices ?? "0",
        method: cfg.method ?? "diffmean",
        token_pos: Number(cfg.token_pos ?? -1),
        normalize: cfg.normalize ?? true,
        positive_samples: (cfg.positive_samples ?? []).length > 0 ? cfg.positive_samples : [""],
        negative_samples: (cfg.negative_samples ?? []).length > 0 ? cfg.negative_samples : [""],
        output_path: cfg.output_path ?? "results/my_vector.gguf",
      };
    } else {
      const cfg =
        scope === "builtin"
          ? builtinTrainingPresets.find((p) => p.name === name)?.config
          : await flask.getTrainingConfig(name);
      if (!cfg) throw new Error(`unknown preset ${name}`);
      training.value = {
        model_path: cfg.model_path ?? "",
        gpu_devices: cfg.gpu_devices ?? "0",
        intervention: cfg.intervention ?? "loreft",
        layer: cfg.reft_config?.layer ?? 8,
        component: cfg.reft_config?.component ?? "block_output",
        low_rank_dimension: cfg.reft_config?.low_rank_dimension ?? 4,
        num_train_epochs: cfg.training_args?.num_train_epochs ?? 100,
        per_device_train_batch_size: cfg.training_args?.per_device_train_batch_size ?? 10,
        learning_rate: cfg.training_args?.learning_rate ?? 0.004,
        logging_steps: cfg.training_args?.logging_steps ?? 40,
        output_dir: cfg.output_dir ?? "results/my_training",
        examples: cfg.training_examples.length > 0 ? cfg.training_examples : [["", ""]],
      };
    }
  } catch (e) {
    presetError.value = (e as Error).message;
  }
}

// ---- Job submission + polling ----
const submitting = ref(false);
const submitError = ref("");
const status = ref<{
  running: boolean;
  message: string;
  error: string | null;
  logs: string[];
  outputPath: string | null;
  extra: string;
} | null>(null);

let pollTimer: ReturnType<typeof setInterval> | null = null;

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

onBeforeUnmount(stopPolling);

async function pollOnce(jobKind: JobKind): Promise<void> {
  try {
    if (jobKind === "extraction") {
      const s = await flask.getExtractionStatus();
      status.value = {
        running: s.is_extracting,
        message: s.status_message,
        error: s.error_message,
        logs: s.logs ?? [],
        outputPath: s.result?.output_path ?? null,
        extra: s.result ? `${t("layers_extracted_label")}: ${s.result.layers_extracted}` : "",
      };
      if (!s.is_extracting && (s.result || s.error_message)) stopPolling();
    } else {
      const s = await flask.getTrainingStatus();
      const done = !s.is_training && (s.status_message.includes("complete") || s.error_message);
      status.value = {
        running: s.is_training,
        message: s.status_message,
        error: s.error_message || null,
        logs: s.logs ?? [],
        // The training pipeline saves ReFT weights into output_dir; the
        // playground consumes them via the pyreft payload adapter.
        outputPath: done && !s.error_message ? training.value.output_dir : null,
        extra: s.is_training ? `epoch ${s.current_epoch ?? 0}, step ${s.current_step ?? 0}` : "",
      };
      if (done) stopPolling();
    }
  } catch (e) {
    status.value = {
      running: false,
      message: "",
      error: (e as Error).message,
      logs: [],
      outputPath: null,
      extra: "",
    };
    stopPolling();
  }
}

function startPolling(jobKind: JobKind): void {
  stopPolling();
  pollTimer = setInterval(() => pollOnce(jobKind), 2000);
  pollOnce(jobKind);
}

async function submitExtraction(): Promise<void> {
  submitting.value = true;
  submitError.value = "";
  try {
    await flask.startExtraction({
      model_path: extraction.value.model_path,
      gpu_devices: extraction.value.gpu_devices,
      method: extraction.value.method,
      token_pos: extraction.value.token_pos,
      normalize: extraction.value.normalize,
      positive_samples: extraction.value.positive_samples.filter((sample) => sample.trim()),
      negative_samples: extraction.value.negative_samples.filter((sample) => sample.trim()),
      output_path: extraction.value.output_path,
    });
    startPolling("extraction");
  } catch (e) {
    submitError.value = (e as Error).message;
  } finally {
    submitting.value = false;
  }
}

async function submitTraining(): Promise<void> {
  submitting.value = true;
  submitError.value = "";
  try {
    await flask.startTraining({
      model_path: training.value.model_path,
      gpu_devices: training.value.gpu_devices,
      intervention: training.value.intervention,
      training_examples: training.value.examples.filter(([a, b]) => a.trim() || b.trim()),
      output_dir: training.value.output_dir,
      reft_config: {
        layer: training.value.layer,
        component: training.value.component,
        low_rank_dimension: training.value.low_rank_dimension,
      },
      training_args: {
        num_train_epochs: training.value.num_train_epochs,
        per_device_train_batch_size: training.value.per_device_train_batch_size,
        learning_rate: training.value.learning_rate,
        logging_steps: training.value.logging_steps,
      },
    });
    startPolling("training");
  } catch (e) {
    submitError.value = (e as Error).message;
  } finally {
    submitting.value = false;
  }
}

const canSubmitExtraction = computed(
  () =>
    extraction.value.model_path.trim() !== "" &&
    extraction.value.positive_samples.some((sample) => sample.trim()) &&
    extraction.value.negative_samples.some((sample) => sample.trim()) &&
    extraction.value.output_path.trim() !== "",
);

const canSubmitTraining = computed(
  () =>
    training.value.model_path.trim() !== "" &&
    training.value.output_dir.trim() !== "" &&
    training.value.examples.some(([a, b]) => a.trim() && b.trim()),
);

function selectKind(next: JobKind): void {
  kind.value = next;
  status.value = null;
  stopPolling();
  refreshPresets();
}

/** Load the produced vector into a fresh playground spec and navigate. */
function useInPlayground(): void {
  if (!status.value?.outputPath) return;
  const spec = defaultSteeringSpec();
  if (kind.value === "extraction") {
    spec.vectors[0].source = status.value.outputPath;
    spec.vectors[0].algorithm = "direct";
  } else {
    // Trained ReFT weights are an in-memory payload server-side; mark
    // the spec so the Python export tells the user to load it via
    // vec.from_pyreft(<output_dir>).
    spec.vectors[0].data = { __inline_payload__: `vec.from_pyreft(${JSON.stringify(status.value.outputPath)})` };
    spec.vectors[0].algorithm = "loreft";
    spec.vectors[0].layers = [training.value.layer];
    spec.vectors[0].apply = {
      ...defaultApplySpec(),
      prompt: null,
      generation: null,
      prompt_positions: [-1],
    };
  }
  loadCustomSpec(spec);
  router.push("/steer");
}

refreshPresets();
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>{{ t("workshop_title") }}</h1>
    </div>
    <p class="page-intro">{{ t("workshop_intro") }}</p>

    <div class="kind-tabs">
      <button
        :class="{ primary: kind === 'extraction' }"
        @click="selectKind('extraction')"
      >
        {{ t("workshop_kind_extraction") }}
      </button>
      <button :class="{ primary: kind === 'training' }" @click="selectKind('training')">
        {{ t("workshop_kind_training") }}
      </button>
    </div>

    <div class="workshop-stack">
      <div class="panel form-panel">
        <div class="field preset-row">
          <label>{{ t("import_config_label") }}</label>
          <div class="preset-controls">
            <select v-model="selectedPreset" class="full">
              <option value="">{{ t("import_config_placeholder") }}</option>
              <optgroup :label="t('presets_builtin_group')">
                <option v-for="p in builtinPresets" :key="p.name" :value="`builtin:${p.name}`">
                  {{ p.display_name }}
                </option>
              </optgroup>
              <optgroup v-if="serverOnlyPresets.length > 0" :label="t('presets_server_group')">
                <option v-for="p in serverOnlyPresets" :key="p.name" :value="`server:${p.name}`">
                  {{ p.display_name ?? p.name }}
                </option>
              </optgroup>
            </select>
            <button class="small" :disabled="!selectedPreset" @click="importPreset">
              {{ t("import_config_label") }}
            </button>
          </div>
          <div v-if="presetError" class="help-text text-err">{{ presetError }}</div>
          <div v-else-if="presetsUnavailable" class="help-text">
            {{ t("presets_unavailable") }}
          </div>
        </div>

        <!-- One shared column grid for both forms, so every box on the
             page lines up on the same edges regardless of field count. -->
        <div class="form-grid">
          <!-- Extraction form -->
          <template v-if="kind === 'extraction'">
            <div class="field span-3">
              <label>{{ t("model_path_label") }}</label>
              <input
                v-model="extraction.model_path"
                type="text"
                class="mono full"
                :placeholder="t('model_path_placeholder')"
              />
              <div class="help-text">{{ t("model_path_help") }}</div>
            </div>
            <div class="field">
              <label>{{ t("gpu_devices_label") }}</label>
              <input
                v-model="extraction.gpu_devices"
                type="text"
                class="mono full"
                :placeholder="t('gpu_devices_placeholder')"
              />
              <div class="help-text">{{ t("gpu_devices_help") }}</div>
            </div>

            <div class="field span-2">
              <label>{{ t("extract_method_label") }}</label>
              <select v-model="extraction.method" class="full">
                <option value="diffmean">{{ t("extract_method_diffmean") }}</option>
                <option value="pca">{{ t("extract_method_pca") }}</option>
                <option value="lat">{{ t("extract_method_lat") }}</option>
              </select>
              <div class="help-text">{{ t("extract_method_help") }}</div>
            </div>
            <div class="field">
              <label>{{ t("extract_token_pos_label") }}</label>
              <input v-model.number="extraction.token_pos" type="number" class="mono full" />
              <div class="help-text">{{ t("extract_token_pos_help") }}</div>
            </div>
            <div class="field check-cell">
              <label class="inline-check">
                <input v-model="extraction.normalize" type="checkbox" />
                {{ t("extract_normalize_label") }}
              </label>
            </div>

            <div class="field span-2">
              <label>{{ t("positive_samples_label") }}</label>
              <StringListEditor
                v-model="extraction.positive_samples"
                :placeholder="t('sample_placeholder')"
              />
              <div class="help-text">{{ t("positive_samples_help") }}</div>
            </div>
            <div class="field span-2">
              <label>{{ t("negative_samples_label") }}</label>
              <StringListEditor
                v-model="extraction.negative_samples"
                :placeholder="t('sample_placeholder')"
              />
              <div class="help-text">{{ t("negative_samples_help") }}</div>
            </div>

            <div class="field span-4">
              <label>{{ t("output_path_label") }}</label>
              <input v-model="extraction.output_path" type="text" class="mono full" />
              <div class="help-text">{{ t("output_path_help") }}</div>
            </div>
            <div class="span-4">
              <button
                class="primary"
                :disabled="submitting || !canSubmitExtraction || status?.running"
                @click="submitExtraction"
              >
                {{ t("start_extraction_btn") }}
              </button>
            </div>
          </template>

          <!-- Training form -->
          <template v-else>
            <div class="field span-3">
              <label>{{ t("model_path_label") }}</label>
              <input
                v-model="training.model_path"
                type="text"
                class="mono full"
                :placeholder="t('model_path_placeholder')"
              />
              <div class="help-text">{{ t("model_path_help") }}</div>
            </div>
            <div class="field">
              <label>{{ t("gpu_devices_label") }}</label>
              <input
                v-model="training.gpu_devices"
                type="text"
                class="mono full"
                :placeholder="t('gpu_devices_placeholder')"
              />
              <div class="help-text">{{ t("gpu_devices_help") }}</div>
            </div>

            <div class="field">
              <label>{{ t("train_intervention_label") }}</label>
              <select v-model="training.intervention" class="full">
                <option value="loreft">loreft</option>
                <option value="bias">bias</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t("train_layer_label") }}</label>
              <input v-model.number="training.layer" type="number" class="mono full" />
            </div>
            <div class="field">
              <label>{{ t("train_component_label") }}</label>
              <select v-model="training.component" class="full">
                <option value="block_output">block_output</option>
                <option value="attention_output">attention_output</option>
                <option value="mlp_output">mlp_output</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t("train_low_rank_dim_label") }}</label>
              <input v-model.number="training.low_rank_dimension" type="number" class="mono full" />
            </div>

            <div class="field">
              <label>{{ t("train_epochs_label") }}</label>
              <input v-model.number="training.num_train_epochs" type="number" class="mono full" />
            </div>
            <div class="field">
              <label>{{ t("train_batch_size_label") }}</label>
              <input
                v-model.number="training.per_device_train_batch_size"
                type="number"
                class="mono full"
              />
            </div>
            <div class="field">
              <label>{{ t("train_learning_rate_label") }}</label>
              <input
                v-model.number="training.learning_rate"
                type="number"
                step="0.0001"
                class="mono full"
              />
            </div>
            <div class="field">
              <label>{{ t("train_logging_steps_label") }}</label>
              <input v-model.number="training.logging_steps" type="number" class="mono full" />
            </div>

            <div class="field span-4">
              <label>{{ t("train_output_dir_label") }}</label>
              <input v-model="training.output_dir" type="text" class="mono full" />
            </div>
            <div class="field span-4">
              <label>{{ t("train_examples_label") }}</label>
              <div class="help-text example-help">{{ t("train_examples_help") }}</div>
              <div class="list-stack">
                <div v-for="(example, i) in training.examples" :key="i" class="list-item">
                  <span class="item-index mono">{{ i + 1 }}</span>
                  <input
                    v-model="example[0]"
                    type="text"
                    class="item-input"
                    :placeholder="t('example_input_placeholder')"
                  />
                  <span class="item-arrow">→</span>
                  <input
                    v-model="example[1]"
                    type="text"
                    class="item-input"
                    :placeholder="t('example_output_placeholder')"
                  />
                  <button
                    class="item-remove"
                    :disabled="training.examples.length <= 1"
                    :title="t('remove_btn')"
                    @click="training.examples.splice(i, 1)"
                  >
                    ✕
                  </button>
                </div>
                <button class="add-btn" @click="training.examples.push(['', ''])">
                  ＋ {{ t("add_example_btn") }}
                </button>
              </div>
            </div>
            <div class="span-4">
              <button
                class="primary"
                :disabled="submitting || !canSubmitTraining || status?.running"
                @click="submitTraining"
              >
                {{ t("start_training_btn") }}
              </button>
            </div>
          </template>
        </div>

        <div v-if="submitError" class="help-text text-err">{{ submitError }}</div>
      </div>

      <!-- Job status + guidance form the right column -->
      <div class="panel status-panel">
        <h2>{{ t("job_status_title") }}</h2>
        <template v-if="status">
          <div class="status-line">
            <span
              class="badge"
              :class="status.error ? 'text-err' : status.running ? 'text-warn' : 'text-ok'"
            >
              {{ status.error ? t("job_failed") : status.running ? t("job_running") : t("job_done") }}
            </span>
            <span v-if="status.extra" class="dim">{{ status.extra }}</span>
          </div>
          <div class="status-message">{{ status.error ?? status.message }}</div>
          <div v-if="status.outputPath" class="result-box">
            <div>
              <span class="dim">{{ t("output_file_label") }}:</span>
              <span class="mono"> {{ status.outputPath }}</span>
            </div>
            <button class="small primary" @click="useInPlayground">
              {{ t("use_in_playground_btn") }}
            </button>
          </div>
          <h3>{{ t("job_logs_title") }}</h3>
          <pre class="code-block logs">{{ status.logs.join("\n") }}</pre>
        </template>
        <p v-else class="dim">{{ t("job_idle") }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kind-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.workshop-stack {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* Four equal columns shared by both forms; fields claim 1-4 of them, so
   inputs on different rows always start and end on the same edges. */
.form-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0 14px;
  align-items: start;
}

.span-2 {
  grid-column: span 2;
}

.span-3 {
  grid-column: span 3;
}

.span-4 {
  grid-column: 1 / -1;
}

/* A lone checkbox has no label above it: nudge it down onto the same
   baseline as the inputs beside it. */
.check-cell {
  align-self: start;
  padding-top: 22px;
}

.preset-controls {
  display: flex;
  gap: 8px;
}

.preset-controls button {
  white-space: nowrap;
}

.example-help {
  margin-bottom: 6px;
}

.status-line {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.status-message {
  font-size: 12.5px;
  margin-bottom: 8px;
  word-break: break-word;
}

.result-box {
  border: 1px solid var(--ok);
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12.5px;
}

.logs {
  max-height: 320px;
  overflow-y: auto;
  font-size: 11.5px;
  white-space: pre-wrap;
}
</style>
