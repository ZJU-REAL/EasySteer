<script setup lang="ts">
/**
 * Compare runner: streams the same prompt twice against the
 * OpenAI-compatible server — baseline (no steering) vs the current spec.
 */
import { computed, ref } from "vue";
import { useI18n } from "../i18n";
import { streamChatCompletion, setServerSteering } from "../lib/openai";
import { playground } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import { validateSteeringSpec, type SteeringSpec } from "../lib/spec";

const props = defineProps<{ spec: SteeringSpec }>();
const { t } = useI18n();

const outputA = ref("");
const outputB = ref("");
const running = ref(false);
const runError = ref("");
const serverDefaultMsg = ref("");

let abort: AbortController | null = null;

const specValid = computed(() => validateSteeringSpec(props.spec).length === 0);

/** `compare=false` runs the steered pane only. */
async function run(compare: boolean): Promise<void> {
  runError.value = "";
  running.value = true;
  outputA.value = "";
  outputB.value = "";
  abort = new AbortController();

  const common = {
    baseUrl: settings.openaiBaseUrl,
    model: settings.model,
    messages: [{ role: "user" as const, content: playground.prompt }],
    temperature: settings.temperature,
    maxTokens: settings.maxTokens,
    signal: abort.signal,
  };

  const runs: Promise<void>[] = [
    streamChatCompletion({
      ...common,
      steering: props.spec,
      onToken: (tok) => (outputB.value += tok),
    }),
  ];
  if (compare) {
    // The baseline pane streams the same prompt with no steering.
    runs.push(
      streamChatCompletion({
        ...common,
        steering: null,
        onToken: (tok) => (outputA.value += tok),
      }),
    );
  }

  try {
    await Promise.all(runs);
  } catch (e) {
    if ((e as Error).name !== "AbortError") {
      runError.value = t("run_error", { error: (e as Error).message });
    }
  } finally {
    running.value = false;
    abort = null;
  }
}

function stop(): void {
  abort?.abort();
}

async function setAsServerDefault(): Promise<void> {
  serverDefaultMsg.value = "";
  try {
    await setServerSteering(settings.openaiBaseUrl, props.spec);
    serverDefaultMsg.value = t("server_default_ok");
  } catch (e) {
    serverDefaultMsg.value = t("run_error", { error: (e as Error).message });
  }
}

</script>

<template>
  <div class="run-panel panel">
    <h2>{{ t("run_title") }}</h2>

    <div class="field">
      <label>{{ t("prompt_label") }}</label>
      <textarea
        v-model="playground.prompt"
        class="full"
        rows="3"
        :placeholder="t('prompt_placeholder')"
      ></textarea>
    </div>

    <div class="field-row sampling-row">
      <div class="field num-field">
        <label>{{ t("temperature_label") }}</label>
        <input v-model.number="settings.temperature" type="number" step="0.1" min="0" class="mono full" />
      </div>
      <div class="field num-field">
        <label>{{ t("max_tokens_label") }}</label>
        <input v-model.number="settings.maxTokens" type="number" min="1" class="mono full" />
      </div>
    </div>

    <div class="button-row">
      <button class="primary" :disabled="running || !specValid || !playground.prompt" @click="run(true)">
        {{ t("run_ab_btn") }}
      </button>
      <button :disabled="running || !specValid || !playground.prompt" @click="run(false)">
        {{ t("run_steered_btn") }}
      </button>
      <button v-if="running" @click="stop">{{ t("stop_btn") }}</button>
      <span class="spacer"></span>
      <button class="small" :disabled="!specValid" @click="setAsServerDefault">
        {{ t("server_default_btn") }}
      </button>
    </div>
    <div v-if="serverDefaultMsg" class="help-text">{{ serverDefaultMsg }}</div>
    <div v-if="runError" class="help-text text-err">{{ runError }}</div>

    <div class="output-grid">
      <div class="output-pane">
        <h3>{{ t("baseline_title") }}</h3>
        <pre class="code-block output-text">{{
          outputA || (running ? t("waiting_stream") : "")
        }}</pre>
      </div>
      <div class="output-pane">
        <h3>{{ t("steered_title") }}</h3>
        <pre class="code-block output-text">{{
          outputB || (running ? t("waiting_stream") : "")
        }}</pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Numeric inputs stay their natural size instead of splitting the row. */
.num-field {
  flex: 0 0 120px !important;
}

.button-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
}

.output-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 6px;
}

.output-text {
  min-height: 140px;
  max-height: 380px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
