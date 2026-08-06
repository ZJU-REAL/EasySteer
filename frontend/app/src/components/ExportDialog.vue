<script setup lang="ts">
/** Modal displaying exported code (Python / extra_body / curl) with copy. */
import { ref } from "vue";
import { useI18n } from "../i18n";

defineProps<{ title: string; code: string }>();

const emit = defineEmits<{ (e: "close"): void }>();
const { t } = useI18n();
const copied = ref(false);

async function copy(code: string): Promise<void> {
  await navigator.clipboard.writeText(code);
  copied.value = true;
  setTimeout(() => (copied.value = false), 1500);
}
</script>

<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="dialog panel">
      <div class="dialog-header">
        <h2>{{ title }}</h2>
        <span class="spacer"></span>
        <button class="small" @click="copy(code)">
          {{ copied ? t("copied") : t("copy_btn") }}
        </button>
        <button class="small" @click="emit('close')">{{ t("close_btn") }}</button>
      </div>
      <pre class="code-block">{{ code }}</pre>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.dialog {
  width: min(760px, 90vw);
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.dialog-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.dialog-header h2 {
  margin: 0;
}

.spacer {
  flex: 1;
}

.code-block {
  overflow: auto;
  flex: 1;
}
</style>
