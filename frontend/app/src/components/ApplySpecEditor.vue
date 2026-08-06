<script setup lang="ts">
/**
 * Form editor for one ApplySpec (mutates the object in place).
 *
 * `phases` is the outer gate. The selection language is fully
 * symmetric — three selectors per phase, each named for it — so the
 * grid shows one row per selector, grouped by phase, with the include
 * and exclude inputs side by side.
 */
import { computed } from "vue";
import { useI18n } from "../i18n";
import type { ApplySpec, Phase } from "../lib/spec";
import IntListInput from "./IntListInput.vue";
import WindowInput from "./WindowInput.vue";

const props = defineProps<{ apply: ApplySpec }>();
const { t } = useI18n();

const excludeCount = computed(
  () =>
    [
      props.apply.exclude_prompt_tokens,
      props.apply.exclude_prompt_positions,
      props.apply.exclude_prompt_window,
      props.apply.exclude_generation_tokens,
      props.apply.exclude_generation_positions,
      props.apply.exclude_generation_window,
    ].filter((v) => v !== null && v !== undefined).length,
);

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

    <div class="selector-grid">
      <div class="head-row">
        <span></span>
        <span class="col-head">{{ t("include_title") }}</span>
        <span class="col-head exclude">
          {{ t("exclusions_title") }}
          <span v-if="excludeCount > 0" class="badge accent">{{ excludeCount }}</span>
        </span>
      </div>

      <div class="group-row" :class="{ 'group-off': !hasPhase('prompt') }">
        <span class="group-name">{{ t("prompt_group_title") }}</span>
        <span class="group-rule"></span>
        <span class="group-rule exclude"></span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('prompt') }">
        <span class="sel-name">
          {{ t("prompt_tokens_label") }}
          <span class="help-text">{{ t("prompt_tokens_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <IntListInput v-model="apply.prompt_tokens" :placeholder="t('tokens_placeholder')" />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <IntListInput
            v-model="apply.exclude_prompt_tokens"
            :placeholder="t('tokens_placeholder')"
          />
        </span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('prompt') }">
        <span class="sel-name">
          {{ t("prompt_positions_label") }}
          <span class="help-text">{{ t("prompt_positions_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <IntListInput
            v-model="apply.prompt_positions"
            :placeholder="t('prompt_positions_placeholder')"
          />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <IntListInput
            v-model="apply.exclude_prompt_positions"
            :placeholder="t('prompt_positions_placeholder')"
          />
        </span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('prompt') }">
        <span class="sel-name">
          {{ t("prompt_window_label") }}
          <span class="help-text">{{ t("prompt_window_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <WindowInput v-model="apply.prompt_window" />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <WindowInput v-model="apply.exclude_prompt_window" />
        </span>
      </div>

      <div class="group-row" :class="{ 'group-off': !hasPhase('generation') }">
        <span class="group-name">{{ t("generation_group_title") }}</span>
        <span class="group-rule"></span>
        <span class="group-rule exclude"></span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('generation') }">
        <span class="sel-name">
          {{ t("generation_tokens_label") }}
          <span class="help-text">{{ t("generation_tokens_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <IntListInput
            v-model="apply.generation_tokens"
            :placeholder="t('tokens_placeholder')"
          />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <IntListInput
            v-model="apply.exclude_generation_tokens"
            :placeholder="t('tokens_placeholder')"
          />
        </span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('generation') }">
        <span class="sel-name">
          {{ t("generation_positions_label") }}
          <span class="help-text">{{ t("generation_positions_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <IntListInput
            v-model="apply.generation_positions"
            :placeholder="t('generation_positions_placeholder')"
          />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <IntListInput
            v-model="apply.exclude_generation_positions"
            :placeholder="t('generation_positions_placeholder')"
          />
        </span>
      </div>

      <div class="sel-row" :class="{ 'sel-off': !hasPhase('generation') }">
        <span class="sel-name">
          {{ t("generation_window_label") }}
          <span class="help-text">{{ t("generation_window_help") }}</span>
        </span>
        <span class="sel-cell">
          <span class="cell-label">{{ t("include_title") }}</span>
          <WindowInput v-model="apply.generation_window" />
        </span>
        <span class="sel-cell exclude">
          <span class="cell-label">{{ t("exclusions_title") }}</span>
          <WindowInput v-model="apply.exclude_generation_window" />
        </span>
      </div>
    </div>

    <div class="help-text">{{ t("exclusions_help") }}</div>
  </fieldset>
</template>

<style scoped>
.apply-editor {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-inset);
  padding: 12px 14px 14px;
  margin: 0;
  min-inline-size: 0;
}

legend {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-dim);
  padding: 0 6px;
}

.phase-row {
  display: flex;
  gap: 16px;
}

/* One row per selector, its include and exclude inputs side by side,
   grouped under a small prompt/generation heading. */
.selector-grid {
  display: grid;
  grid-template-columns: minmax(170px, 0.95fr) 1fr 1fr;
  column-gap: 14px;
  row-gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}

.head-row,
.sel-row,
.group-row {
  display: contents;
}

.col-head {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 6px;
  align-self: end;
  padding-bottom: 2px;
}

.group-name {
  font-size: 11px;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  margin-top: 2px;
}

.group-rule {
  border-top: 1px solid var(--border);
  align-self: center;
}

/* Rows of an unchecked phase stay editable but read as inactive. */
.group-off .group-name {
  color: var(--text-dim);
}

.sel-off {
  opacity: 0.55;
}

.sel-name {
  display: flex;
  flex-direction: column;
  font-size: 12px;
  font-weight: 500;
}

.sel-name .help-text {
  font-weight: 400;
  margin-top: 1px;
}

.sel-cell {
  display: block;
  min-width: 0;
}

/* The exclude column is set off by a dashed rule so both halves stay
   comparable at a glance; the rule is drawn on every cell so it runs
   the height of the column. */
.sel-cell.exclude,
.col-head.exclude,
.group-rule.exclude {
  border-left: 1px dashed var(--border-strong);
  padding-left: 14px;
}

.sel-cell.exclude {
  align-self: stretch;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.cell-label {
  display: none;
  font-size: 11px;
  color: var(--text-dim);
  margin-bottom: 2px;
}

@media (max-width: 860px) {
  .selector-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .head-row {
    display: none;
  }

  .group-row {
    display: block;
  }

  .group-row .group-rule {
    display: none;
  }

  .sel-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 10px;
  }

  .sel-name {
    grid-column: 1 / -1;
  }

  .cell-label {
    display: block;
  }

  .sel-cell.exclude {
    border-left: none;
    padding-left: 0;
  }
}
</style>
