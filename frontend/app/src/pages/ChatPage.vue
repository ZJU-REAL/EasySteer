<script setup lang="ts">
/**
 * Multi-turn streaming chat over the OpenAI-compatible route.
 *
 * Layout follows the legacy chat page: a configuration rail on the left
 * (presets, steering spec, sampling) and the conversation on the right,
 * shown as a single pane or as baseline/steered panes side by side when
 * comparison is on.
 */
import { computed, nextTick, ref, watch } from "vue";
import { useRouter } from "vue-router";
import AppIcon from "../components/AppIcon.vue";
import SettingsBar from "../components/SettingsBar.vue";
import { useI18n } from "../i18n";
import { chatPresets, type ChatPreset } from "../data/chatPresets";
import { modelShortName } from "../data/gallery";
import { chat, clearChat, type ChatTurn } from "../lib/chatStore";
import { streamChatCompletion, type ChatMessage } from "../lib/openai";
import { playground } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import {
  specFromJson,
  specToJson,
  validateSteeringSpec,
  type SteeringSpec,
} from "../lib/spec";

type PaneKey = "steered" | "baseline";

const router = useRouter();
const { t } = useI18n();

const draft = ref("");
const running = ref(false);
const runError = ref("");
const viewsEl = ref<HTMLElement | null>(null);

let abort: AbortController | null = null;

const playgroundSummary = computed(() => {
  const spec = playground.spec;
  const algos = [...new Set(spec.vectors.map((v) => v.algorithm))].join(", ");
  const sources = spec.vectors.map((v) => v.source ?? "(inline data)").join(", ");
  return `${spec.vectors.length} vector(s), ${algos} — ${sources}`;
});

/** A preset is "active" while the custom spec still matches it verbatim. */
const activePresetId = computed(() => {
  if (chat.steeringMode !== "custom") return null;
  const text = chat.customSpecText.trim();
  return chatPresets.find((p) => JSON.stringify(p.spec, null, 2) === text)?.id ?? null;
});

const compareView = computed(() => chat.compare && chat.steeringMode !== "none");

const panes = computed<PaneKey[]>(() =>
  compareView.value
    ? ["baseline", "steered"]
    : [chat.steeringMode === "none" ? "baseline" : "steered"],
);

function paneText(turn: ChatTurn, pane: PaneKey): string {
  if (turn.role === "user") return turn.content;
  if (pane === "steered") return turn.content;
  if (turn.baseline !== undefined) return turn.baseline;
  // A steered-only turn has no baseline half to show.
  return turn.steered ? "" : turn.content;
}

/**
 * The active steering choice, parsed and validated once: the Send gate,
 * the send path and the inline error all read this. `spec` is null when
 * steering is off or the choice is not usable yet.
 */
const steering = computed<{ ready: boolean; spec: SteeringSpec | null; error: string }>(() => {
  if (chat.steeringMode === "none") return { ready: true, spec: null, error: "" };
  if (chat.steeringMode === "playground") {
    const ok = validateSteeringSpec(playground.spec).length === 0;
    return { ready: ok, spec: ok ? playground.spec : null, error: "" };
  }
  if (chat.customSpecText.trim() === "") return { ready: false, spec: null, error: "" };
  try {
    const spec = specFromJson(JSON.parse(chat.customSpecText));
    const issues = validateSteeringSpec(spec);
    if (issues.length > 0) {
      return {
        ready: false,
        spec: null,
        error: issues.map((i) => `${i.path}: ${i.message}`).join("; "),
      };
    }
    return { ready: true, spec, error: "" };
  } catch (e) {
    return { ready: false, spec: null, error: (e as Error).message };
  }
});

function applyPreset(preset: ChatPreset): void {
  chat.steeringMode = "custom";
  chat.customSpecText = JSON.stringify(preset.spec, null, 2);
}

function seedCustomFromPlayground(): void {
  chat.customSpecText = JSON.stringify(specToJson(playground.spec), null, 2);
}

async function scrollToBottom(): Promise<void> {
  await nextTick();
  for (const list of viewsEl.value?.querySelectorAll(".chat-messages") ?? []) {
    list.scrollTo({ top: list.scrollHeight });
  }
}

watch(
  () => chat.turns.length,
  () => void scrollToBottom(),
);

async function send(): Promise<void> {
  const text = draft.value.trim();
  if (!text || running.value || !steering.value.ready) return;
  const spec = steering.value.spec;

  const comparing = chat.compare && spec !== null;
  runError.value = "";
  draft.value = "";
  chat.turns.push({ role: "user", content: text });
  const reply = reactiveReply(spec !== null, comparing);

  const messages: ChatMessage[] = [];
  for (const turn of chat.turns.slice(0, -1)) {
    messages.push({ role: turn.role, content: turn.content });
  }

  running.value = true;
  abort = new AbortController();

  const stream = (steeringSpec: SteeringSpec | null, sink: (tok: string) => void) =>
    streamChatCompletion({
      baseUrl: settings.openaiBaseUrl,
      model: settings.model,
      messages,
      steering: steeringSpec,
      temperature: settings.temperature,
      maxTokens: settings.maxTokens,
      signal: abort!.signal,
      onToken: (tok) => {
        sink(tok);
        void scrollToBottom();
      },
    });

  try {
    const runs = [stream(spec, (tok) => (reply.content += tok))];
    if (comparing) {
      // The baseline reply streams concurrently with no steering attached.
      runs.push(stream(null, (tok) => (reply.baseline = (reply.baseline ?? "") + tok)));
    }
    const results = await Promise.allSettled(runs);
    const failure = results.find(
      (r): r is PromiseRejectedResult =>
        r.status === "rejected" && (r.reason as Error).name !== "AbortError",
    );
    if (failure) {
      runError.value = t("run_error", { error: (failure.reason as Error).message });
    }
  } finally {
    if (reply.content === "" && !reply.baseline) {
      // Drop the empty assistant turn so the transcript stays clean.
      chat.turns.splice(chat.turns.indexOf(reply), 1);
    }
    running.value = false;
    abort = null;
  }
}

/** Push the assistant placeholder turn and return the reactive object. */
function reactiveReply(steered: boolean, comparing: boolean) {
  const turn = comparing
    ? { role: "assistant" as const, content: "", steered, baseline: "" }
    : { role: "assistant" as const, content: "", steered };
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

    <!-- One settings row: steering choice and sampling side by side, with
         the spec editor unfolding below only when it has something to
         show, so nothing sits in an empty box. -->
    <div class="panel chat-settings">
      <div class="field-row settings-row">
        <div class="field steering-field">
          <label>{{ t("steering_panel_title") }}</label>
          <select v-model="chat.steeringMode" class="full">
            <option value="none">{{ t("steering_mode_none") }}</option>
            <option value="playground">{{ t("steering_mode_playground") }}</option>
            <option value="custom">{{ t("steering_mode_custom") }}</option>
          </select>
        </div>
        <div class="field num-field">
          <label>{{ t("temperature_label") }}</label>
          <input
            v-model.number="settings.temperature"
            type="number"
            min="0"
            max="2"
            step="0.1"
            class="mono full"
          />
        </div>
        <div class="field num-field">
          <label>{{ t("max_tokens_label") }}</label>
          <input v-model.number="settings.maxTokens" type="number" min="1" class="mono full" />
        </div>
        <div class="field check-field">
          <label class="inline-check" :title="t('chat_compare_help')">
            <input
              v-model="chat.compare"
              type="checkbox"
              :disabled="chat.steeringMode === 'none'"
            />
            {{ t("chat_compare_label") }}
          </label>
        </div>
      </div>

      <div v-if="chat.steeringMode === 'playground'" class="spec-row">
        <span class="help-text">
          {{ t("playground_spec_summary", { summary: playgroundSummary }) }}
        </span>
        <button class="small" @click="router.push('/steer')">
          {{ t("edit_in_playground_btn") }}
        </button>
      </div>
      <div v-else-if="chat.steeringMode === 'custom'" class="spec-block">
        <div class="spec-head">
          <label>{{ t("spec_json_title") }}</label>
          <span class="spacer"></span>
          <button class="small" @click="seedCustomFromPlayground">
            {{ t("chat_seed_from_steer_btn") }}
          </button>
        </div>
        <textarea
          v-model="chat.customSpecText"
          class="mono full custom-spec"
          rows="5"
          spellcheck="false"
        ></textarea>
        <div v-if="steering.error" class="help-text text-err">{{ steering.error }}</div>
      </div>
      <div v-else class="help-text">{{ t("chat_steering_help") }}</div>
    </div>

    <div class="chat-layout">
      <aside class="panel preset-panel">
        <h2 class="section-title">{{ t("chat_presets_label") }}</h2>
        <div class="preset-list">
          <button
            v-for="preset in chatPresets"
            :key="preset.id"
            class="preset-item"
            :class="{ active: activePresetId === preset.id }"
            @click="applyPreset(preset)"
          >
            <span class="preset-icon">{{ preset.icon }}</span>
            <span class="preset-text">
              <span class="preset-name">{{ preset.label[settings.language] }}</span>
              <span class="preset-desc dim">{{ preset.description[settings.language] }}</span>
              <span class="preset-model mono dim">{{ modelShortName(preset.model) }}</span>
            </span>
          </button>
        </div>
        <div class="help-text preset-note">{{ t("chat_presets_help") }}</div>
      </aside>

      <section class="chat-main">
        <div v-if="chat.turns.length === 0" class="panel chat-views empty-panel">
          <div class="empty-state">
            <AppIcon name="chat" :size="34" />
            <p>{{ t("chat_empty") }}</p>
            <p class="help-text">{{ t("chat_empty_hint") }}</p>
          </div>
        </div>
        <div v-else ref="viewsEl" class="chat-views" :class="{ split: panes.length > 1 }">
          <div v-for="pane in panes" :key="pane" class="chat-view panel">
            <div class="chat-view-header" :class="pane">
              {{ pane === "steered" ? t("chat_compare_steered") : t("chat_compare_baseline") }}
            </div>
            <div class="chat-messages">
              <div v-for="(turn, i) in chat.turns" :key="i" class="turn" :class="turn.role">
                <span class="turn-role dim">
                  {{ turn.role === "user" ? t("chat_role_user") : t("chat_role_assistant") }}
                </span>
                <div class="turn-content">
                  <template v-if="paneText(turn, pane)">{{ paneText(turn, pane) }}</template>
                  <span
                    v-else-if="running && turn.role === 'assistant' && i === chat.turns.length - 1"
                    class="dim"
                    >{{ t("waiting_stream") }}</span
                  >
                  <span v-else-if="turn.role === 'assistant'" class="dim">{{
                    t("chat_no_reply")
                  }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-if="runError" class="help-text text-err">{{ runError }}</div>

        <div class="composer panel">
          <textarea
            v-model="draft"
            class="full composer-input"
            rows="2"
            :placeholder="t('chat_input_placeholder')"
            @keydown="onDraftKeydown"
          ></textarea>
          <button
            class="primary"
            :disabled="running || !draft.trim() || !steering.ready"
            @click="send"
          >
            {{ t("send_btn") }}
          </button>
          <button v-if="running" @click="stop">{{ t("stop_btn") }}</button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.chat-settings {
  margin-bottom: 14px;
  padding: 10px 12px;
}

.settings-row {
  align-items: flex-end;
}

.settings-row .field {
  margin-bottom: 0;
}

.steering-field {
  flex: 0 0 240px !important;
}

.num-field {
  flex: 0 0 110px !important;
}

.check-field {
  flex: 0 0 auto !important;
  padding-bottom: 7px;
}

.spec-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
}

.spec-block {
  margin-top: 8px;
}

.spec-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.spec-head label {
  margin-bottom: 0;
}

/* The page fills the scroll area and the conversation row takes whatever
   is left, so opening the spec editor shrinks the transcript instead of
   pushing the composer off screen. Both columns share that one row, so
   the preset rail ends on exactly the same line as the composer. */
.chat-page {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.chat-layout {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  gap: 14px;
  flex: 1;
  min-height: 430px;
}

.preset-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.preset-panel .section-title {
  margin-bottom: 8px;
}

.preset-list {
  flex: 0 1 auto;
  min-height: 0;
  overflow-y: auto;
}

.preset-note {
  padding-top: 8px;
}

.preset-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.preset-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  text-align: left;
  border-radius: var(--radius-sm);
  background: var(--bg-inset);
  border-color: var(--border);
  box-shadow: none;
}

.preset-item.active {
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
  background: var(--accent-soft);
}

.preset-icon {
  font-size: 19px;
  line-height: 1;
  flex-shrink: 0;
}

.preset-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.preset-name {
  font-size: 12.5px;
  font-weight: 600;
}

.preset-desc {
  font-size: 11px;
  line-height: 1.4;
}

.preset-model {
  font-size: 10px;
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.custom-spec {
  min-height: 96px;
}

/* Right column: toolbar, conversation panes, composer — a fixed-height
   stack so the composer stays put while the panes scroll. */
.chat-main {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
}

.chat-views {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}

.chat-views.split {
  grid-template-columns: 1fr 1fr;
}

.empty-panel {
  display: flex;
  align-items: center;
  justify-content: center;
}

.empty-panel .empty-state {
  max-width: 46ch;
}

.empty-panel .empty-state p {
  margin: 0;
}

.chat-view {
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 0;
  overflow: hidden;
}

.chat-view-header {
  padding: 7px 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  background: var(--bg-inset);
  border-radius: var(--radius) var(--radius) 0 0;
}

.chat-view-header.steered {
  color: var(--accent);
}

.chat-messages {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.turn {
  max-width: 88%;
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

.turn-role {
  font-size: 11px;
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
  padding: 10px 12px;
  flex-shrink: 0;
}

.composer-input {
  resize: none;
}

.composer button {
  flex-shrink: 0;
}

@media (max-width: 980px) {
  .chat-layout {
    grid-template-columns: 1fr;
  }

  .chat-main {
    position: static;
    height: auto;
  }

  .chat-views {
    min-height: 380px;
  }

  .chat-views.split {
    grid-template-columns: 1fr;
  }
}
</style>
