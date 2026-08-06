<script setup lang="ts">
/**
 * Live SteeringSpec JSON, two-way: reflects the form state and accepts
 * direct edits (a structurally valid edit replaces the spec; invalid
 * JSON shows an error and keeps the last good spec).
 */
import { computed, ref, watch } from "vue";
import { useI18n } from "../i18n";
import {
  specFromJson,
  specToJson,
  validateSteeringSpec,
  type SteeringSpec,
} from "../lib/spec";

const props = defineProps<{ spec: SteeringSpec; revision: number }>();

const emit = defineEmits<{
  (e: "replace", spec: SteeringSpec): void;
}>();

const { t } = useI18n();

const text = ref(render(props.spec));
const parseError = ref("");
const editing = ref(false);

function render(spec: SteeringSpec): string {
  return JSON.stringify(specToJson(spec), null, 2);
}

watch(
  [() => props.spec, () => props.revision],
  () => {
    // While the user is typing here, their own edits round-trip through
    // `replace`; re-rendering would clobber cursor position/formatting.
    if (editing.value) return;
    text.value = render(props.spec);
    parseError.value = "";
  },
  { deep: true },
);

function onInput(): void {
  try {
    const parsed = JSON.parse(text.value);
    const spec = specFromJson(parsed);
    parseError.value = "";
    emit("replace", spec);
  } catch (e) {
    parseError.value = (e as Error).message;
  }
}

function onBlur(): void {
  editing.value = false;
  if (!parseError.value) text.value = render(props.spec);
}

const issues = computed(() => validateSteeringSpec(props.spec));
</script>

<template>
  <div class="json-panel">
    <div class="json-header">
      <h3>{{ t("spec_json_title") }}</h3>
      <span
        v-if="!parseError"
        class="badge"
        :class="issues.length === 0 ? 'text-ok' : 'text-warn'"
      >
        {{ issues.length === 0 ? t("validation_ok") : t("validation_issues", { n: issues.length }) }}
      </span>
    </div>
    <textarea
      v-model="text"
      class="mono json-text"
      spellcheck="false"
      @focus="editing = true"
      @blur="onBlur"
      @input="onInput"
    ></textarea>
    <div v-if="parseError" class="help-text text-err">
      {{ t("json_parse_error", { error: parseError }) }}
    </div>
    <ul v-if="issues.length > 0" class="issue-list">
      <li v-for="(issue, i) in issues" :key="i" class="text-warn">
        <span class="mono">{{ issue.path }}</span
        >: {{ issue.message }}
      </li>
    </ul>
    <div class="help-text">{{ t("spec_json_help") }}</div>
  </div>
</template>

<style scoped>
.json-panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.json-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.json-header h3 {
  margin: 0;
}

.json-text {
  width: 100%;
  min-height: 340px;
  flex: 1;
  background: var(--bg-inset);
  line-height: 1.45;
  white-space: pre;
}

.issue-list {
  margin: 0;
  padding-left: 18px;
  font-size: 12px;
}
</style>
