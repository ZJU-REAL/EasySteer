<script setup lang="ts">
/** Modal shell: overlay, head/body/foot slots, Escape and backdrop close. */
import { onMounted, onUnmounted } from "vue";
import { useI18n } from "../i18n";

withDefaults(defineProps<{ title: string; width?: string }>(), { width: "760px" });

const emit = defineEmits<{ (e: "close"): void }>();
const { t } = useI18n();

function onKey(event: KeyboardEvent): void {
  if (event.key === "Escape") emit("close");
}

onMounted(() => window.addEventListener("keydown", onKey));
onUnmounted(() => window.removeEventListener("keydown", onKey));
</script>

<template>
  <div class="modal-overlay" @click.self="emit('close')">
    <div class="modal" :style="{ width: `min(${width}, 100%)` }">
      <div class="modal-head">
        <h2>{{ title }}</h2>
        <span class="spacer"></span>
        <slot name="head-actions"></slot>
        <button class="small ghost" :title="t('close_btn')" @click="emit('close')">✕</button>
      </div>
      <div class="modal-body">
        <slot></slot>
      </div>
      <div v-if="$slots.footer" class="modal-foot">
        <slot name="footer"></slot>
      </div>
    </div>
  </div>
</template>
