<script setup lang="ts">
/** Renders one icon from the shared path table. */
import { ICON_PATHS, type IconName } from "../lib/icons";

withDefaults(defineProps<{ name: IconName; size?: number }>(), { size: 17 });
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.7"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path v-for="(d, i) in ICON_PATHS[name]" :key="i" :d="d" />
  </svg>
</template>
