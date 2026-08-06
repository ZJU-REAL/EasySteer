<script setup lang="ts">
/**
 * Form editor for one ApplySpec (mutates the object in place).
 *
 * Each phase is selected independently: the "entire phase" toggle is
 * the widest include selector (`prompt="all"` / `generation="all"`),
 * and the three narrower selectors per phase union with it. A phase
 * with nothing set is untouched. The grid shows one row per selector,
 * grouped by phase, include and exclude inputs side by side.
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

function isAll(phase: Phase): boolean {
  return props.apply[phase] === "all";
}

function toggleAll(phase: Phase): void {
  props.apply[phase] = props.apply[phase] === "all" ? null : "all";
}

/** A phase is covered when "all" is on or one of its includes is set. */
function coversPhase(phase: Phase): boolean {
  if (props.apply[phase] === "all") return true;
  const keys =
    phase === "prompt"
      ? (["prompt_tokens", "prompt_positions", "prompt_window"] as const)
      : ([
          "generation_tokens",
          "generation_positions",
          "generation_window",
        ] as const);
  return keys.some((k) => props.apply[k] !== null && props.apply[k] !== undefined);
}
</script>

<template>
  <fieldset class="apply-editor">
    <legend>{{ t("apply_title") }}</legend>
    <div class="help-text intro-help">{{ t("selectors_help") }}</div>

    <div class="selector-grid">
      <div class="head-row">
        <span></span>
        <span class="col-head">{{ t("include_title") }}</span>
        <span class="col-head exclude">
          {{ t("exclusions_title") }}
          <span v-if="excludeCount > 0" class="badge accent">{{ excludeCount }}</span>
        </span>
      </div>

      <div class="group-row" :class="{ 'group-off': !coversPhase('prompt') }">
        <span class="group-name">
          {{ t("prompt_group_title") }}
          <label class="inline-check all-check">
            <input type="checkbox" :checked="isAll('prompt')" @change="toggleAll('prompt')" />
            {{ t("phase_all_label") }}
          </label>
        </span>
        <span class="group-rule"></span>
        <span class="group-rule exclude"></span>
      </div>

      <div class="sel-row">
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

      <div class="sel-row">
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

      <div class="sel-row">
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

      <div class="group-row" :class="{ 'group-off': !coversPhase('generation') }">
        <span class="group-name">
          {{ t("generation_group_title") }}
          <label class="inline-check all-check">
            <input
              type="checkbox"
              :checked="isAll('generation')"
              @change="toggleAll('generation')"
            />
            {{ t("phase_all_label") }}
          </label>
        </span>
        <span class="group-rule"></span>
        <span class="group-rule exclude"></span>
      </div>

      <div class="sel-row">
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

      <div class="sel-row">
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

      <div class="sel-row">
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
.intro-help {
  margin: 0 0 10px;
}

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
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 11px;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  margin-top: 2px;
}

/* The whole-phase toggle rides on the group heading. */
.all-check {
  font-size: 11.5px;
  font-weight: 500;
  letter-spacing: normal;
  text-transform: none;
}

.group-rule {
  border-top: 1px solid var(--border);
  align-self: center;
}

/* An uncovered phase (no "all", no selector) reads as untouched. */
.group-off .group-name {
  color: var(--text-dim);
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
