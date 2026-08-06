<script setup lang="ts">
/**
 * Spec-transparent playground: a full-width builder (vectors behind
 * tabs) above the live SteeringSpec JSON (two-way, collapsible) with the
 * export actions in the JSON panel header, and an A/B compare runner
 * streaming against the OpenAI-compatible server.
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
  <div class="page">
    <div class="page-header">
      <h1>{{ t("playground_title") }}</h1>
      <template v-if="preset">
        <span class="badge accent">{{ preset.method }}</span>
        <span class="badge mono">{{ preset.model }}</span>
      </template>
      <span class="spacer"></span>
      <button class="small" @click="resetPlayground">{{ t("reset_btn") }}</button>
    </div>
    <p class="page-intro">{{ t("steer_intro") }}</p>

    <SettingsBar />

    <div class="panel builder-panel">
      <div class="panel-header">
        <h2>{{ t("spec_builder_title") }}</h2>
      </div>
      <SpecBuilder :key="playground.revision" :spec="playground.spec" />
    </div>

    <SpecJsonPanel
      class="json-block"
      :spec="playground.spec"
      :revision="playground.revision"
      @replace="onJsonReplace"
    >
      <template #actions>
        <span class="dim export-label">{{ t("export_title") }}</span>
        <button class="small" @click="exportPython">{{ t("export_python_btn") }}</button>
        <button class="small" @click="exportExtraBody">{{ t("export_extra_body_btn") }}</button>
        <button class="small" @click="exportCurl">{{ t("export_curl_btn") }}</button>
      </template>
    </SpecJsonPanel>

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
.builder-panel {
  margin-bottom: 14px;
}

.json-block {
  margin-bottom: 14px;
}

.export-label {
  font-size: 11.5px;
}
</style>
