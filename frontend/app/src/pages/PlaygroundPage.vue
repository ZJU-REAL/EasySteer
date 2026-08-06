<script setup lang="ts">
/**
 * Spec-transparent playground: form-driven builder with the live
 * SteeringSpec JSON side by side (two-way), export buttons, and an A/B
 * compare runner streaming against the OpenAI-compatible server.
 */
import { computed, ref } from "vue";
import ExportDialog from "../components/ExportDialog.vue";
import RunPanel from "../components/RunPanel.vue";
import SettingsBar from "../components/SettingsBar.vue";
import SpecBuilder from "../components/SpecBuilder.vue";
import SpecJsonPanel from "../components/SpecJsonPanel.vue";
import { getGalleryEntry } from "../data/gallery";
import { useI18n } from "../i18n";
import { toCurl, toExtraBodyJson, toPython } from "../lib/exporters";
import { playground, replaceSpec, resetPlayground } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import type { SteeringSpec } from "../lib/spec";

const { t } = useI18n();

const preset = computed(() =>
  playground.presetId ? getGalleryEntry(playground.presetId) : undefined,
);

const exportTitle = ref("");
const exportCode = ref("");

function exportPython(): void {
  exportTitle.value = t("export_python_btn");
  exportCode.value = toPython(playground.spec, {
    model: playground.presetModel || settings.model || undefined,
    prompt: playground.prompt || undefined,
    maxTokens: settings.maxTokens,
    temperature: settings.temperature,
  });
}

function exportExtraBody(): void {
  exportTitle.value = t("export_extra_body_btn");
  exportCode.value = toExtraBodyJson(playground.spec);
}

function exportCurl(): void {
  exportTitle.value = t("export_curl_btn");
  exportCode.value = toCurl(playground.spec, {
    baseUrl: settings.openaiBaseUrl,
    model: settings.model || playground.presetModel || undefined,
    prompt: playground.prompt || undefined,
  });
}

function onJsonReplace(spec: SteeringSpec): void {
  replaceSpec(spec);
}
</script>

<template>
  <div>
    <div class="page-header">
      <h1>{{ t("playground_title") }}</h1>
      <template v-if="preset">
        <span class="badge accent">{{ preset.method }}</span>
        <span class="badge mono">{{ preset.model }}</span>
      </template>
      <span class="spacer"></span>
      <button class="small" @click="resetPlayground">{{ t("reset_btn") }}</button>
    </div>

    <SettingsBar />

    <div class="builder-grid">
      <div class="builder-col">
        <h2>{{ t("spec_builder_title") }}</h2>
        <SpecBuilder :key="playground.revision" :spec="playground.spec" />
      </div>
      <div class="json-col">
        <SpecJsonPanel
          :spec="playground.spec"
          :revision="playground.revision"
          @replace="onJsonReplace"
        />
        <div class="export-row">
          <span class="dim">{{ t("export_title") }}:</span>
          <button class="small" @click="exportPython">{{ t("export_python_btn") }}</button>
          <button class="small" @click="exportExtraBody">{{ t("export_extra_body_btn") }}</button>
          <button class="small" @click="exportCurl">{{ t("export_curl_btn") }}</button>
        </div>
      </div>
    </div>

    <RunPanel :spec="playground.spec" />

    <ExportDialog
      v-if="exportCode"
      :title="exportTitle"
      :code="exportCode"
      @close="exportCode = ''"
    />
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.page-header h1 {
  margin: 0;
}

.spacer {
  flex: 1;
}

.builder-grid {
  display: grid;
  grid-template-columns: minmax(380px, 1fr) minmax(320px, 1fr);
  gap: 14px;
  margin-bottom: 14px;
  align-items: start;
}

@media (max-width: 1100px) {
  .builder-grid {
    grid-template-columns: 1fr;
  }
}

.export-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
</style>
