<script setup lang="ts">
/**
 * Multi-turn streaming chat over the OpenAI-compatible route, with an
 * optional steering spec attached to every assistant reply — either the
 * current playground spec or a custom JSON spec edited inline.
 */
import { computed, nextTick, ref, watch } from "vue";
import { useRouter } from "vue-router";
import SettingsBar from "../components/SettingsBar.vue";
import { useI18n } from "../i18n";
import { chat, clearChat } from "../lib/chatStore";
import { streamChatCompletion, type ChatMessage } from "../lib/openai";
import { playground } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import {
  specFromJson,
  specToJson,
  validateSteeringSpec,
  type SteeringSpec,
} from "../lib/spec";

const router = useRouter();
const { t } = useI18n();

const draft = ref("");
const running = ref(false);
const runError = ref("");
const customSpecError = ref("");
const transcript = ref<HTMLElement | null>(null);

let abort: AbortController | null = null;

const showSteeringPanel = ref(chat.steeringMode !== "none");

const playgroundSummary = computed(() => {
  const spec = playground.spec;
  const algos = [...new Set(spec.vectors.map((v) => v.algorithm))].join(", ");
  const sources = spec.vectors
    .map((v) => v.source ?? "(inline data)")
    .join(", ");
  return `${spec.vectors.length} vector(s), ${algos} — ${sources}`;
});

function resolveSteering(): SteeringSpec | null {
  if (chat.steeringMode === "none") return null;
  if (chat.steeringMode === "playground") {
    return validateSteeringSpec(playground.spec).length === 0 ? playground.spec : null;
  }
  try {
    const spec = specFromJson(JSON.parse(chat.customSpecText));
    const issues = validateSteeringSpec(spec);
    if (issues.length > 0) {
      customSpecError.value = issues.map((i) => `${i.path}: ${i.message}`).join("; ");
      return null;
    }
    customSpecError.value = "";
    return spec;
  } catch (e) {
    customSpecError.value = (e as Error).message;
    return null;
  }
}

const steeringReady = computed(() => {
  if (chat.steeringMode === "none") return true;
  if (chat.steeringMode === "playground") {
    return validateSteeringSpec(playground.spec).length === 0;
  }
  if (chat.customSpecText.trim() === "") return false;
  try {
    return validateSteeringSpec(specFromJson(JSON.parse(chat.customSpecText))).length === 0;
  } catch {
    return false;
  }
});

function seedCustomFromPlayground(): void {
  chat.customSpecText = JSON.stringify(specToJson(playground.spec), null, 2);
  customSpecError.value = "";
}

async function scrollToBottom(): Promise<void> {
  await nextTick();
  transcript.value?.scrollTo({ top: transcript.value.scrollHeight });
}

watch(
  () => chat.turns.length,
  () => void scrollToBottom(),
);

async function send(): Promise<void> {
  const text = draft.value.trim();
  if (!text || running.value || !steeringReady.value) return;
  const steering = resolveSteering();
  if (chat.steeringMode !== "none" && steering === null) return;

  runError.value = "";
  draft.value = "";
  chat.turns.push({ role: "user", content: text });
  const reply = reactiveReply(steering !== null);

  const messages: ChatMessage[] = [];
  if (chat.systemPrompt.trim()) {
    messages.push({ role: "system", content: chat.systemPrompt });
  }
  for (const turn of chat.turns.slice(0, -1)) {
    messages.push({ role: turn.role, content: turn.content });
  }

  running.value = true;
  abort = new AbortController();
  try {
    await streamChatCompletion({
      baseUrl: settings.openaiBaseUrl,
      model: settings.model,
      messages,
      steering,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      signal: abort.signal,
      onToken: (tok) => {
        reply.content += tok;
        void scrollToBottom();
      },
    });
  } catch (e) {
    if ((e as Error).name !== "AbortError") {
      runError.value = t("run_error", { error: (e as Error).message });
    }
    if (reply.content === "") {
      // Drop the empty assistant turn so the transcript stays clean.
      chat.turns.splice(chat.turns.indexOf(reply), 1);
    }
  } finally {
    running.value = false;
    abort = null;
  }
}

/** Push the assistant placeholder turn and return the reactive object. */
function reactiveReply(steered: boolean) {
  const turn = { role: "assistant" as const, content: "", steered };
  chat.turns.push(turn);
  return chat.turns[chat.turns.length - 1];
}

function stop(): void {
  abort?.abort();
}

function onDraftKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void send();
  }
}
</script>

<template>
  <div class="page chat-page">
    <div class="page-header">
      <h1>{{ t("chat_title") }}</h1>
      <span class="spacer"></span>
      <button class="small" :disabled="chat.turns.length === 0" @click="clearChat">
        {{ t("clear_chat_btn") }}
      </button>
    </div>
    <p class="page-intro">{{ t("chat_intro") }}</p>

    <SettingsBar />

    <div class="panel steering-panel">
      <button type="button" class="small" @click="showSteeringPanel = !showSteeringPanel">
        {{ showSteeringPanel ? "▾" : "▸" }} {{ t("steering_panel_title") }}
        <span v-if="chat.steeringMode !== 'none'" class="badge accent">
          {{ t("steering_active_badge") }}
        </span>
      </button>
      <div v-show="showSteeringPanel" class="steering-body">
        <div class="field-row">
          <div class="field">
            <label>{{ t("steering_panel_title") }}</label>
            <select v-model="chat.steeringMode" class="full">
              <option value="none">{{ t("steering_mode_none") }}</option>
              <option value="playground">{{ t("steering_mode_playground") }}</option>
              <option value="custom">{{ t("steering_mode_custom") }}</option>
            </select>
          </div>
          <div class="field">
            <label>{{ t("system_prompt_label") }}</label>
            <input v-model="chat.systemPrompt" type="text" class="full" />
          </div>
        </div>
        <template v-if="chat.steeringMode === 'playground'">
          <div class="help-text">
            {{ t("playground_spec_summary", { summary: playgroundSummary }) }}
          </div>
          <button class="small" @click="router.push('/playground')">
            {{ t("edit_in_playground_btn") }}
          </button>
        </template>
        <template v-else-if="chat.steeringMode === 'custom'">
          <div class="field">
            <label>{{ t("spec_json_title") }}</label>
            <textarea
              v-model="chat.customSpecText"
              class="mono full custom-spec"
              rows="8"
              spellcheck="false"
            ></textarea>
            <div v-if="customSpecError" class="help-text text-err">{{ customSpecError }}</div>
          </div>
          <button class="small" @click="seedCustomFromPlayground">
            {{ t("steering_mode_playground") }} →
          </button>
        </template>
      </div>
    </div>

    <div ref="transcript" class="transcript panel">
      <p v-if="chat.turns.length === 0" class="dim empty-hint">{{ t("chat_empty") }}</p>
      <div
        v-for="(turn, i) in chat.turns"
        :key="i"
        class="turn"
        :class="turn.role"
      >
        <div class="turn-meta">
          <span class="turn-role">{{
            turn.role === "user" ? t("chat_role_user") : t("chat_role_assistant")
          }}</span>
          <span v-if="turn.steered" class="badge accent">{{ t("steering_active_badge") }}</span>
        </div>
        <div class="turn-content">{{ turn.content || (running && i === chat.turns.length - 1 ? t("waiting_stream") : "") }}</div>
      </div>
    </div>

    <div v-if="runError" class="help-text text-err">{{ runError }}</div>

    <div class="composer">
      <textarea
        v-model="draft"
        class="full composer-input"
        rows="2"
        :placeholder="t('chat_input_placeholder')"
        @keydown="onDraftKeydown"
      ></textarea>
      <div class="composer-actions">
        <button class="primary" :disabled="running || !draft.trim() || !steeringReady" @click="send">
          {{ t("send_btn") }}
        </button>
        <button v-if="running" @click="stop">{{ t("stop_btn") }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-page {
  max-width: 900px;
  display: flex;
  flex-direction: column;
}

.steering-panel {
  margin-bottom: 12px;
  padding: 8px 12px;
}

.steering-body {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
}

.steering-body .field-row {
  width: 100%;
}

.custom-spec {
  min-height: 120px;
}

.transcript {
  flex: 1;
  min-height: 320px;
  max-height: 55vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 10px;
}

.empty-hint {
  margin: auto;
  text-align: center;
}

.turn {
  max-width: 85%;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.turn.user {
  align-self: flex-end;
  align-items: flex-end;
}

.turn.assistant {
  align-self: flex-start;
}

.turn-meta {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 11px;
  color: var(--text-dim);
}

.turn-content {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 12px;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg-inset);
}

.turn.user .turn-content {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.composer {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.composer-input {
  resize: none;
}

.composer-actions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
