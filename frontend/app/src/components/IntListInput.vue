<script setup lang="ts">
/**
 * Text input bound to a `number[] | null` model using the compact
 * "0-27" / "16,17,18" syntax. Invalid text shows an inline error and
 * leaves the model untouched.
 */
import { ref, watch } from "vue";
import { formatIntList, parseIntListString } from "../lib/spec";

const props = defineProps<{
  modelValue: number[] | null;
  placeholder?: string;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: number[] | null): void;
}>();

const text = ref(formatIntList(props.modelValue));
const error = ref("");

watch(
  () => props.modelValue,
  (value) => {
    // Re-sync from outside only when the value genuinely differs from
    // what the current text parses to (avoids clobbering "1,2," mid-edit).
    try {
      const current = parseIntListString(text.value);
      if (JSON.stringify(current) === JSON.stringify(value)) return;
    } catch {
      // Local text is invalid; external value wins.
    }
    text.value = formatIntList(value);
    error.value = "";
  },
);

function onInput(): void {
  try {
    const parsed = parseIntListString(text.value);
    error.value = "";
    emit("update:modelValue", parsed);
  } catch (e) {
    error.value = (e as Error).message;
  }
}
</script>

<template>
  <div>
    <input
      v-model="text"
      class="mono list-input"
      type="text"
      :placeholder="placeholder"
      @input="onInput"
    />
    <div v-if="error" class="help-text text-err">{{ error }}</div>
  </div>
</template>

<style scoped>
.list-input {
  width: 100%;
}
</style>
