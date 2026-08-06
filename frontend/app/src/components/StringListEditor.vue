<script setup lang="ts">
/**
 * Add-item editor for a list of text samples. Each row is one enclosed
 * control: an index cap on the left, the text field in the middle and a
 * remove button on the right, so numbers and ✕ buttons line up as
 * columns. Pasting multi-line text into a row splits it into one item
 * per line, so bulk entry still works.
 */
import { useI18n } from "../i18n";

const props = defineProps<{ modelValue: string[]; placeholder?: string }>();

const emit = defineEmits<{
  (e: "update:modelValue", value: string[]): void;
}>();

const { t } = useI18n();

function setItem(index: number, value: string): void {
  const next = [...props.modelValue];
  next[index] = value;
  emit("update:modelValue", next);
}

function addItem(): void {
  emit("update:modelValue", [...props.modelValue, ""]);
}

function removeItem(index: number): void {
  const next = props.modelValue.filter((_, i) => i !== index);
  emit("update:modelValue", next.length > 0 ? next : [""]);
}

function onPaste(index: number, event: ClipboardEvent): void {
  const text = event.clipboardData?.getData("text") ?? "";
  if (!text.includes("\n")) return;
  event.preventDefault();
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  const next = [...props.modelValue];
  next.splice(index, 1, ...(lines.length > 0 ? lines : [""]));
  emit("update:modelValue", next);
}
</script>

<template>
  <div class="list-stack">
    <div v-for="(item, i) in modelValue" :key="i" class="list-item">
      <span class="item-index mono">{{ i + 1 }}</span>
      <input
        :value="item"
        type="text"
        class="item-input"
        :placeholder="props.placeholder"
        @input="setItem(i, ($event.target as HTMLInputElement).value)"
        @paste="onPaste(i, $event)"
      />
      <button
        type="button"
        class="item-remove"
        :disabled="modelValue.length === 1 && modelValue[0] === ''"
        :title="t('remove_btn')"
        @click="removeItem(i)"
      >
        ✕
      </button>
    </div>
    <button type="button" class="add-btn" @click="addItem">＋ {{ t("add_item_btn") }}</button>
  </div>
</template>

<!-- Row styling comes from the shared list-row primitives in style.css. -->
