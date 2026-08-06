<script setup lang="ts">
/**
 * Editor for a half-open [start, stop] window bound to
 * `SpecWindow | null`. Clearing the start clears the whole window;
 * clearing the stop makes it open-ended (stop=null).
 */
import { computed } from "vue";
import { useI18n } from "../i18n";
import type { SpecWindow } from "../lib/spec";

const props = defineProps<{ modelValue: SpecWindow | null }>();

const emit = defineEmits<{
  (e: "update:modelValue", value: SpecWindow | null): void;
}>();

const { t } = useI18n();

function asInt(value: unknown): number | null {
  // v-model.number yields "" for a cleared input.
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isInteger(n) ? n : null;
}

const start = computed({
  get: () => props.modelValue?.[0] ?? null,
  set: (value: unknown) => {
    const n = asInt(value);
    if (n === null) {
      emit("update:modelValue", null);
    } else {
      emit("update:modelValue", [n, props.modelValue?.[1] ?? null]);
    }
  },
});

const stop = computed({
  get: () => props.modelValue?.[1] ?? null,
  set: (value: unknown) => {
    const n = asInt(value);
    if (props.modelValue === null) {
      if (n !== null) emit("update:modelValue", [0, n]);
      return;
    }
    emit("update:modelValue", [props.modelValue[0], n]);
  },
});
</script>

<template>
  <div class="window-row">
    <input
      v-model.number="start"
      type="number"
      class="mono window-input"
      :placeholder="t('window_start_placeholder')"
    />
    <span class="dim">..</span>
    <input
      v-model.number="stop"
      type="number"
      class="mono window-input"
      :placeholder="t('window_stop_placeholder')"
    />
  </div>
</template>

<style scoped>
.window-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.window-input {
  width: 110px;
}
</style>
