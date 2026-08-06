<script setup lang="ts">
/** Form editor for one VectorSpec (mutates the object in place). */
import { computed, ref, watch } from "vue";
import { useI18n } from "../i18n";
import { ALGORITHMS, type VectorSpec } from "../lib/spec";
import ApplySpecEditor from "./ApplySpecEditor.vue";
import IntListInput from "./IntListInput.vue";

const props = defineProps<{ vector: VectorSpec; index: number; removable: boolean }>();

const emit = defineEmits<{
  (e: "remove"): void;
  (e: "duplicate"): void;
}>();

const { t } = useI18n();

const hasInlineData = computed(
  () => props.vector.data !== null && props.vector.data !== undefined,
);

const sourceModel = computed({
  get: () => props.vector.source ?? "",
  set: (value: string) => {
    props.vector.source = value === "" ? null : value;
  },
});

const nameModel = computed({
  get: () => props.vector.name ?? "",
  set: (value: string) => {
    props.vector.name = value === "" ? null : value;
  },
});

// Params edited as JSON text (only moe_router takes params today).
const showParams = computed(
  () => props.vector.algorithm === "moe_router" || Object.keys(props.vector.params).length > 0,
);
const paramsText = ref(JSON.stringify(props.vector.params));
const paramsError = ref("");

watch(
  () => props.vector.params,
  (value) => {
    try {
      if (JSON.stringify(JSON.parse(paramsText.value)) === JSON.stringify(value)) return;
    } catch {
      // Local text invalid: external value wins.
    }
    paramsText.value = JSON.stringify(value);
    paramsError.value = "";
  },
);

function onParamsInput(): void {
  try {
    const parsed = JSON.parse(paramsText.value);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("params must be a JSON object");
    }
    paramsError.value = "";
    props.vector.params = parsed;
  } catch (e) {
    paramsError.value = (e as Error).message;
  }
}
</script>

<template>
  <div class="vector-editor">
    <div class="vector-header">
      <h3>{{ t("vector_n_title", { n: index + 1 }) }}</h3>
      <span v-if="vector.algorithm !== 'direct'" class="badge accent mono">{{
        vector.algorithm
      }}</span>
      <span class="spacer"></span>
      <button class="small" @click="emit('duplicate')">{{ t("duplicate_vector_btn") }}</button>
      <button v-if="removable" class="small" @click="emit('remove')">
        {{ t("remove_vector_btn") }}
      </button>
    </div>

    <div v-if="hasInlineData" class="inline-data-notice">
      {{ t("data_inline_notice") }}
    </div>
    <div v-else class="field">
      <label>{{ t("source_label") }}</label>
      <input
        v-model="sourceModel"
        type="text"
        class="mono full"
        :placeholder="t('source_placeholder')"
      />
      <div class="help-text">{{ t("source_help") }}</div>
    </div>

    <div class="field-row">
      <div class="field">
        <label>{{ t("algorithm_label") }}</label>
        <select v-model="vector.algorithm" class="mono full">
          <option v-for="algo in ALGORITHMS" :key="algo" :value="algo">{{ algo }}</option>
        </select>
      </div>
      <div class="field">
        <label>{{ t("scale_label") }}</label>
        <input v-model.number="vector.scale" type="number" step="0.1" class="mono full" />
      </div>
    </div>

    <div class="field-row">
      <div class="field">
        <label>{{ t("layers_label") }}</label>
        <IntListInput v-model="vector.layers" :placeholder="t('layers_placeholder')" />
      </div>
      <div class="field">
        <label>{{ t("name_label") }}</label>
        <input v-model="nameModel" type="text" class="full" />
      </div>
    </div>

    <div class="field">
      <label class="inline-check">
        <input v-model="vector.normalize" type="checkbox" />
        {{ t("normalize_label") }}
      </label>
    </div>

    <div v-if="showParams" class="field">
      <label>{{ t("params_label") }}</label>
      <input v-model="paramsText" type="text" class="mono full" @input="onParamsInput" />
      <div v-if="paramsError" class="help-text text-err">{{ paramsError }}</div>
    </div>

    <ApplySpecEditor :apply="vector.apply" />
  </div>
</template>

<style scoped>
.vector-editor {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--bg-panel);
}

.vector-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.vector-header h3 {
  margin: 0;
}

.spacer {
  flex: 1;
}

.full {
  width: 100%;
}

.inline-check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text);
  font-size: 12.5px;
}

.inline-data-notice {
  background: var(--accent-soft);
  border: 1px dashed var(--accent);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  margin-bottom: 8px;
}
</style>
