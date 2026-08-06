<script setup lang="ts">
/** Exported code (Python / extra_body / curl) in a modal, with copy. */
import { ref } from "vue";
import ModalDialog from "./ModalDialog.vue";
import { useI18n } from "../i18n";

const props = defineProps<{ title: string; code: string }>();

const emit = defineEmits<{ (e: "close"): void }>();
const { t } = useI18n();
const copied = ref(false);

async function copy(): Promise<void> {
  await navigator.clipboard.writeText(props.code);
  copied.value = true;
  setTimeout(() => (copied.value = false), 1500);
}
</script>

<template>
  <ModalDialog :title="title" @close="emit('close')">
    <template #head-actions>
      <button class="small" @click="copy">{{ copied ? t("copied") : t("copy_btn") }}</button>
    </template>
    <pre class="code-block">{{ code }}</pre>
  </ModalDialog>
</template>
