<script setup lang="ts">
/** Connection settings for the OpenAI-compatible inference server. */
import { ref } from "vue";
import { useI18n } from "../i18n";
import { listModels } from "../lib/openai";
import { settings } from "../lib/settings";

const { t } = useI18n();

const checkResult = ref("");
const checkOk = ref(false);

async function checkConnection(): Promise<void> {
  checkResult.value = "...";
  try {
    const models = await listModels(settings.openaiBaseUrl);
    checkResult.value = t("connection_ok", { models: models.join(", ") || "-" });
    if (!settings.model && models.length > 0) settings.model = models[0];
    checkOk.value = true;
  } catch (e) {
    checkOk.value = false;
    checkResult.value = t("connection_failed", { error: (e as Error).message });
  }
}
</script>

<template>
  <div class="panel settings-bar">
    <div class="field-row settings-row">
      <div class="field grow2">
        <label>{{ t("openai_base_url_label") }}</label>
        <input v-model="settings.openaiBaseUrl" type="text" class="mono full" />
      </div>
      <div class="field">
        <label>{{ t("model_label") }}</label>
        <input
          v-model="settings.model"
          type="text"
          class="mono full"
          :placeholder="t('model_placeholder')"
        />
      </div>
      <div class="field check-field">
        <button class="small" @click="checkConnection">{{ t("check_connection_btn") }}</button>
      </div>
    </div>
    <div class="help-text">{{ t("openai_base_url_help") }}</div>
    <div v-if="checkResult" class="help-text" :class="checkOk ? 'text-ok' : 'text-err'">
      {{ checkResult }}
    </div>
  </div>
</template>

<style scoped>
.settings-bar {
  margin-bottom: 14px;
  padding: 10px 12px;
}

.grow2 {
  flex: 2 !important;
}

.settings-row {
  align-items: flex-end;
}

.check-field {
  flex: 0 0 auto !important;
}

.check-field button {
  padding: 5px 10px;
  white-space: nowrap;
}

.field {
  margin-bottom: 0;
}
</style>
