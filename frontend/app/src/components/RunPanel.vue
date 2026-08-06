<script setup lang="ts">
/**
 * A/B compare runner: streams the same prompt twice against the
 * OpenAI-compatible server — baseline (no steering) vs the current spec,
 * or spec A vs a second spec B provided as JSON.
 */
import { computed, ref } from "vue";
import { useI18n } from "../i18n";
import { streamChatCompletion, setServerSteering, type ChatMessage } from "../lib/openai";
import { playground } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import { specFromJson, validateSteeringSpec, type SteeringSpec } from "../lib/spec";

const props = defineProps<{ spec: SteeringSpec }>();
const { t } = useI18n();

const abMode = ref<"baseline" | "two_specs">("baseline");
const specBText = ref("");
const specBError = ref("");

const outputA = ref("");
const outputB = ref("");
const running = ref(false);
const runError = ref("");
const serverDefaultMsg = ref("");

let abort: AbortController | null = null;

const specValid = computed(() => validateSteeringSpec(props.spec).length === 0);

function parseSpecB(): SteeringSpec | null {
  try {
    const spec = specFromJson(JSON.parse(specBText.value));
    const issues = validateSteeringSpec(spec);
    if (issues.length > 0) {
      specBError.value = issues.map((i) => `${i.path}: ${i.message}`).join("; ");
      return null;
    }
    specBError.value = "";
    return spec;
  } catch (e) {
    specBError.value = (e as Error).message;
    return null;
  }
}

function buildMessages(): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (playground.systemPrompt.trim()) {
    messages.push({ role: "system", content: playground.systemPrompt });
  }
  messages.push({ role: "user", content: playground.prompt });
  return messages;
}

async function run(compare: boolean): Promise<void> {
  runError.value = "";
  // Pane A holds the baseline (or spec A); pane B holds the steered run.
  let steeringA: SteeringSpec | null = null;
  const streamA = compare;
  let steeringB: SteeringSpec | null = props.spec;
  if (abMode.value === "two_specs" && compare) {
    steeringA = props.spec;
    steeringB = parseSpecB();
    if (steeringB === null) return;
  }

  running.value = true;
  outputA.value = "";
  outputB.value = "";
  abort = new AbortController();
  const messages = buildMessages();

  const common = {
    baseUrl: settings.openaiBaseUrl,
    model: settings.model,
    messages,
    temperature: settings.temperature,
    maxTokens: settings.maxTokens,
    signal: abort.signal,
  };

  const runs: Promise<void>[] = [
    streamChatCompletion({
      ...common,
      steering: steeringB,
      onToken: (tok) => (outputB.value += tok),
    }),
  ];
  if (streamA) {
    runs.push(
      streamChatCompletion({
        ...common,
        steering: steeringA,
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

const titleA = computed(() =>
  abMode.value === "two_specs" ? t("spec_a_title") : t("baseline_title"),
);
const titleB = computed(() =>
  abMode.value === "two_specs" ? t("spec_b_title") : t("steered_title"),
);
</script>

<template>
  <div class="run-panel panel">
    <h2>{{ t("run_title") }}</h2>

    <div class="field">
      <label>{{ t("system_prompt_label") }}</label>
      <input v-model="playground.systemPrompt" type="text" class="full" />
    </div>
    <div class="field">
      <label>{{ t("prompt_label") }}</label>
      <textarea
        v-model="playground.prompt"
        class="full"
        rows="3"
        :placeholder="t('prompt_placeholder')"
      ></textarea>
    </div>

    <div class="field-row">
      <div class="field">
        <label>{{ t("temperature_label") }}</label>
        <input v-model.number="settings.temperature" type="number" step="0.1" min="0" class="mono full" />
      </div>
      <div class="field">
        <label>{{ t("max_tokens_label") }}</label>
        <input v-model.number="settings.maxTokens" type="number" min="1" class="mono full" />
      </div>
      <div class="field">
        <label>{{ t("ab_mode_label") }}</label>
        <select v-model="abMode" class="full">
          <option value="baseline">{{ t("ab_mode_baseline") }}</option>
          <option value="two_specs">{{ t("ab_mode_two_specs") }}</option>
        </select>
      </div>
    </div>

    <div v-if="abMode === 'two_specs'" class="field">
      <label>{{ t("spec_b_title") }}</label>
      <textarea v-model="specBText" class="mono full" rows="5" spellcheck="false"></textarea>
      <div v-if="specBError" class="help-text text-err">{{ specBError }}</div>
    </div>

    <div class="button-row">
      <button class="primary" :disabled="running || !specValid || !playground.prompt" @click="run(true)">
        {{ t("run_ab_btn") }}
      </button>
      <button
        v-if="abMode === 'baseline'"
        :disabled="running || !specValid || !playground.prompt"
        @click="run(false)"
      >
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
        <h3>{{ titleA }}</h3>
        <pre class="code-block output-text">{{
          outputA || (running ? t("waiting_stream") : "")
        }}</pre>
      </div>
      <div class="output-pane">
        <h3>{{ titleB }}</h3>
        <pre class="code-block output-text">{{
          outputB || (running ? t("waiting_stream") : "")
        }}</pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.full {
  width: 100%;
}

.button-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
}

.spacer {
  flex: 1;
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
