<script setup lang="ts">
/**
 * Vector workshop: extraction and training merged into one flow.
 * Pick a method, configure, submit the job to the retained Flask backend,
 * poll status, then hand the produced vector to the playground.
 */
import { computed, onBeforeUnmount, ref } from "vue";
import { useRouter } from "vue-router";
import SettingsBar from "../components/SettingsBar.vue";
import { useI18n } from "../i18n";
import * as flask from "../lib/flask";
import { playground, replaceSpec } from "../lib/playgroundStore";
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
  positive_text: "",
  negative_text: "",
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
const presets = ref<{ name: string; display_name?: string }[]>([]);
const selectedPreset = ref("");
const presetError = ref("");

async function refreshPresets(): Promise<void> {
  presetError.value = "";
  selectedPreset.value = "";
  try {
    const resp =
      kind.value === "extraction"
        ? await flask.listExtractionConfigs()
        : await flask.listTrainingConfigs();
    presets.value = resp.configs;
  } catch (e) {
    presets.value = [];
    presetError.value = (e as Error).message;
  }
}

async function importPreset(): Promise<void> {
  if (!selectedPreset.value) return;
  presetError.value = "";
  try {
    if (kind.value === "extraction") {
      const cfg = await flask.getExtractionConfig(selectedPreset.value);
      extraction.value = {
        model_path: cfg.model_path ?? "",
        gpu_devices: cfg.gpu_devices ?? "0",
        method: cfg.method ?? "diffmean",
        token_pos: Number(cfg.token_pos ?? -1),
        normalize: cfg.normalize ?? true,
        positive_text: (cfg.positive_samples ?? []).join("\n"),
        negative_text: (cfg.negative_samples ?? []).join("\n"),
        output_path: cfg.output_path ?? "results/my_vector.gguf",
      };
    } else {
      const cfg = await flask.getTrainingConfig(selectedPreset.value);
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
        output_dir:
          (cfg as { output_dir?: string }).output_dir ??
          cfg.training_args?.["output_dir" as never] ??
          "results/my_training",
        examples:
          (cfg.training_examples as [string, string][] | undefined)?.length
            ? (cfg.training_examples as [string, string][])
            : [["", ""]],
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
      positive_samples: extraction.value.positive_text.split("\n").filter((s) => s.trim()),
      negative_samples: extraction.value.negative_text.split("\n").filter((s) => s.trim()),
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
    extraction.value.positive_text.trim() !== "" &&
    extraction.value.negative_text.trim() !== "" &&
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
      phases: ["prompt"],
      positions: [-1],
    };
  }
  replaceSpec(spec);
  playground.presetId = null;
  playground.presetModel = "";
  router.push("/playground");
}

refreshPresets();
</script>

<template>
  <div>
    <h1>{{ t("workshop_title") }}</h1>
    <p class="dim intro">{{ t("workshop_intro") }}</p>

    <SettingsBar :show-flask="true" />

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

    <div class="workshop-grid">
      <div class="panel form-panel">
        <div class="field preset-row">
          <label>{{ t("import_config_label") }}</label>
          <div class="preset-controls">
            <select v-model="selectedPreset" class="full">
              <option value="">{{ t("import_config_placeholder") }}</option>
              <option v-for="p in presets" :key="p.name" :value="p.name">
                {{ p.display_name ?? p.name }}
              </option>
            </select>
            <button class="small" :disabled="!selectedPreset" @click="importPreset">
              {{ t("import_config_label") }}
            </button>
          </div>
          <div v-if="presetError" class="help-text text-err">{{ presetError }}</div>
        </div>

        <!-- Extraction form -->
        <template v-if="kind === 'extraction'">
          <div class="field-row">
            <div class="field">
              <label>{{ t("model_path_label") }}</label>
              <input
                v-model="extraction.model_path"
                type="text"
                class="mono full"
                :placeholder="t('model_path_placeholder')"
              />
            </div>
            <div class="field gpu-field">
              <label>{{ t("gpu_devices_label") }}</label>
              <input
                v-model="extraction.gpu_devices"
                type="text"
                class="mono full"
                :placeholder="t('gpu_devices_placeholder')"
              />
            </div>
          </div>
          <div class="field-row">
            <div class="field">
              <label>{{ t("extract_method_label") }}</label>
              <select v-model="extraction.method" class="full">
                <option value="diffmean">{{ t("extract_method_diffmean") }}</option>
                <option value="pca">{{ t("extract_method_pca") }}</option>
                <option value="lat">{{ t("extract_method_lat") }}</option>
              </select>
            </div>
            <div class="field">
              <label>{{ t("extract_token_pos_label") }}</label>
              <input v-model.number="extraction.token_pos" type="number" class="mono full" />
              <div class="help-text">{{ t("extract_token_pos_help") }}</div>
            </div>
            <div class="field normalize-field">
              <label class="inline-check">
                <input v-model="extraction.normalize" type="checkbox" />
                {{ t("extract_normalize_label") }}
              </label>
            </div>
          </div>
          <div class="field">
            <label>{{ t("positive_samples_label") }}</label>
            <textarea v-model="extraction.positive_text" class="full" rows="5"></textarea>
            <div class="help-text">{{ t("positive_samples_help") }}</div>
          </div>
          <div class="field">
            <label>{{ t("negative_samples_label") }}</label>
            <textarea v-model="extraction.negative_text" class="full" rows="5"></textarea>
            <div class="help-text">{{ t("negative_samples_help") }}</div>
          </div>
          <div class="field">
            <label>{{ t("output_path_label") }}</label>
            <input v-model="extraction.output_path" type="text" class="mono full" />
            <div class="help-text">{{ t("output_path_help") }}</div>
          </div>
          <button
            class="primary"
            :disabled="submitting || !canSubmitExtraction || status?.running"
            @click="submitExtraction"
          >
            {{ t("start_extraction_btn") }}
          </button>
        </template>

        <!-- Training form -->
        <template v-else>
          <div class="field-row">
            <div class="field">
              <label>{{ t("model_path_label") }}</label>
              <input
                v-model="training.model_path"
                type="text"
                class="mono full"
                :placeholder="t('model_path_placeholder')"
              />
            </div>
            <div class="field gpu-field">
              <label>{{ t("gpu_devices_label") }}</label>
              <input v-model="training.gpu_devices" type="text" class="mono full" />
            </div>
          </div>
          <div class="field-row">
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
          </div>
          <div class="field-row">
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
          </div>
          <div class="field">
            <label>{{ t("train_output_dir_label") }}</label>
            <input v-model="training.output_dir" type="text" class="mono full" />
          </div>
          <div class="field">
            <label>{{ t("train_examples_label") }}</label>
            <div class="help-text">{{ t("train_examples_help") }}</div>
            <div v-for="(example, i) in training.examples" :key="i" class="example-row">
              <input
                v-model="example[0]"
                type="text"
                class="full"
                :placeholder="t('example_input_placeholder')"
              />
              <input
                v-model="example[1]"
                type="text"
                class="full"
                :placeholder="t('example_output_placeholder')"
              />
              <button
                class="small"
                :disabled="training.examples.length <= 1"
                @click="training.examples.splice(i, 1)"
              >
                x
              </button>
            </div>
            <button class="small" @click="training.examples.push(['', ''])">
              + {{ t("add_example_btn") }}
            </button>
          </div>
          <button
            class="primary"
            :disabled="submitting || !canSubmitTraining || status?.running"
            @click="submitTraining"
          >
            {{ t("start_training_btn") }}
          </button>
        </template>

        <div v-if="submitError" class="help-text text-err">{{ submitError }}</div>
      </div>

      <!-- Job status -->
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
.intro {
  margin-top: 0;
  max-width: 70ch;
}

.kind-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.workshop-grid {
  display: grid;
  grid-template-columns: minmax(420px, 3fr) minmax(300px, 2fr);
  gap: 14px;
  align-items: start;
}

@media (max-width: 1100px) {
  .workshop-grid {
    grid-template-columns: 1fr;
  }
}

.full {
  width: 100%;
}

.gpu-field {
  flex: 0 0 160px !important;
}

.normalize-field {
  flex: 0 0 auto !important;
  align-self: end;
  padding-bottom: 22px;
}

.inline-check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text);
  font-size: 12.5px;
}

.preset-controls {
  display: flex;
  gap: 8px;
}

.example-row {
  display: flex;
  gap: 6px;
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
