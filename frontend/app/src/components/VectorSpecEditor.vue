<script setup lang="ts">
/** Form editor for one VectorSpec (mutates the object in place). */
import { computed, ref, watch } from "vue";
import { useI18n } from "../i18n";
import { ALGORITHMS, type VectorSpec } from "../lib/spec";
import ApplySpecEditor from "./ApplySpecEditor.vue";
import IntListInput from "./IntListInput.vue";

const props = defineProps<{ vector: VectorSpec }>();

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

    <div class="field-row scalar-row">
      <div class="field">
        <label>{{ t("algorithm_label") }}</label>
        <select v-model="vector.algorithm" class="mono full">
          <option v-for="algo in ALGORITHMS" :key="algo" :value="algo">{{ algo }}</option>
        </select>
      </div>
      <div class="field scale-field">
        <label>{{ t("scale_label") }}</label>
        <div class="scale-row">
          <input v-model.number="vector.scale" type="range" min="-5" max="5" step="0.1" />
          <input v-model.number="vector.scale" type="number" step="0.1" class="mono scale-num" />
        </div>
      </div>
      <div class="field">
        <label>{{ t("layers_label") }}</label>
        <IntListInput v-model="vector.layers" :placeholder="t('layers_placeholder')" />
      </div>
      <div class="field normalize-field">
        <label class="inline-check">
          <input v-model="vector.normalize" type="checkbox" />
          {{ t("normalize_label") }}
        </label>
      </div>
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
  display: flex;
  flex-direction: column;
}

.scalar-row {
  flex-wrap: wrap;
  align-items: flex-end;
}

.scalar-row > .field {
  flex: 1 1 190px;
}

.scale-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.scale-row input[type="range"] {
  flex: 1;
  min-width: 0;
  padding: 0;
  border: none;
  background: transparent;
  accent-color: var(--accent);
  box-shadow: none;
}

.scale-num {
  width: 74px;
  flex-shrink: 0;
}

.normalize-field {
  flex: 0 0 auto !important;
  padding-bottom: 7px;
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
