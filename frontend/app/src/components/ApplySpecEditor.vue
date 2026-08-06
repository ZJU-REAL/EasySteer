<script setup lang="ts">
/**
 * Form editor for one ApplySpec (mutates the object in place).
 *
 * `phases` is the outer gate; the five include selectors union within
 * it, and the symmetric exclude selectors (collapsible section) union
 * and always subtract — exclusion wins on overlap.
 */
import { computed, ref } from "vue";
import { useI18n } from "../i18n";
import type { ApplySpec, Phase } from "../lib/spec";
import IntListInput from "./IntListInput.vue";
import WindowInput from "./WindowInput.vue";

const props = defineProps<{ apply: ApplySpec }>();
const { t } = useI18n();

const excludeCount = computed(
  () =>
    [
      props.apply.exclude_tokens,
      props.apply.exclude_positions,
      props.apply.exclude_prompt_window,
      props.apply.exclude_generation_positions,
      props.apply.exclude_generation_window,
    ].filter((v) => v !== null && v !== undefined).length,
);

// Auto-expand when the spec (e.g. a gallery preset) carries exclusions.
const showExcludes = ref(excludeCount.value > 0);

function hasPhase(phase: Phase): boolean {
  return props.apply.phases.includes(phase);
}

function togglePhase(phase: Phase): void {
  if (hasPhase(phase)) {
    props.apply.phases = props.apply.phases.filter((p) => p !== phase);
  } else {
    // Keep canonical prompt-first order.
    const next = [...props.apply.phases, phase];
    props.apply.phases = (["prompt", "generation"] as Phase[]).filter((p) =>
      next.includes(p),
    );
  }
}
</script>

<template>
  <fieldset class="apply-editor">
    <legend>{{ t("apply_title") }}</legend>
    <div class="field">
      <label>{{ t("phases_label") }}</label>
      <div class="phase-row">
        <label class="inline-check">
          <input type="checkbox" :checked="hasPhase('prompt')" @change="togglePhase('prompt')" />
          {{ t("phase_prompt") }}
        </label>
        <label class="inline-check">
          <input
            type="checkbox"
            :checked="hasPhase('generation')"
            @change="togglePhase('generation')"
          />
          {{ t("phase_generation") }}
        </label>
      </div>
      <div class="help-text">{{ t("selectors_help") }}</div>
    </div>

    <div class="field-row">
      <div class="field">
        <label>{{ t("tokens_label") }}</label>
        <IntListInput v-model="apply.tokens" :placeholder="t('tokens_placeholder')" />
      </div>
      <div class="field">
        <label>{{ t("positions_label") }}</label>
        <IntListInput v-model="apply.positions" :placeholder="t('positions_placeholder')" />
      </div>
    </div>

    <div class="field-row">
      <div class="field">
        <label>{{ t("prompt_window_label") }}</label>
        <WindowInput v-model="apply.prompt_window" />
        <div class="help-text">{{ t("prompt_window_help") }}</div>
      </div>
      <div class="field">
        <label>{{ t("generation_window_label") }}</label>
        <WindowInput v-model="apply.generation_window" />
        <div class="help-text">{{ t("generation_window_help") }}</div>
      </div>
    </div>

    <div class="field">
      <label>{{ t("generation_positions_label") }}</label>
      <IntListInput
        v-model="apply.generation_positions"
        :placeholder="t('generation_positions_placeholder')"
      />
    </div>

    <button type="button" class="small excludes-toggle" @click="showExcludes = !showExcludes">
      {{ showExcludes ? "▾" : "▸" }} {{ t("exclusions_title") }}
      <span v-if="excludeCount > 0" class="badge accent">{{ excludeCount }}</span>
    </button>
    <div v-show="showExcludes" class="excludes-section">
      <div class="help-text">{{ t("exclusions_help") }}</div>
      <div class="field-row">
        <div class="field">
          <label>{{ t("exclude_tokens_label") }}</label>
          <IntListInput v-model="apply.exclude_tokens" />
        </div>
        <div class="field">
          <label>{{ t("exclude_positions_label") }}</label>
          <IntListInput v-model="apply.exclude_positions" />
        </div>
      </div>
      <div class="field-row">
        <div class="field">
          <label>{{ t("exclude_prompt_window_label") }}</label>
          <WindowInput v-model="apply.exclude_prompt_window" />
        </div>
        <div class="field">
          <label>{{ t("exclude_generation_window_label") }}</label>
          <WindowInput v-model="apply.exclude_generation_window" />
        </div>
      </div>
      <div class="field">
        <label>{{ t("exclude_generation_positions_label") }}</label>
        <IntListInput
          v-model="apply.exclude_generation_positions"
          :placeholder="t('generation_positions_placeholder')"
        />
      </div>
    </div>
  </fieldset>
</template>

<style scoped>
.apply-editor {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  margin: 0;
}

legend {
  font-size: 12px;
  color: var(--text-dim);
  padding: 0 4px;
}

.phase-row {
  display: flex;
  gap: 16px;
}

.excludes-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
}

.excludes-section {
  margin-top: 8px;
  padding: 8px 10px;
  border: 1px dashed var(--border);
  border-radius: 6px;
}
</style>
