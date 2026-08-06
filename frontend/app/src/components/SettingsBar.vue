<script setup lang="ts">
/** Connection settings: OpenAI-compatible base URL, model, Flask URL. */
import { ref } from "vue";
import { useI18n } from "../i18n";
import { listModels } from "../lib/openai";
import { settings } from "../lib/settings";

const props = defineProps<{ showFlask?: boolean }>();
const { t } = useI18n();

const checkResult = ref("");
const checkOk = ref(false);

async function checkConnection(): Promise<void> {
  checkResult.value = "...";
  try {
    const models = await listModels(settings.openaiBaseUrl);
    checkOk.value = true;
    checkResult.value = t("connection_ok", { models: models.join(", ") || "-" });
    if (!settings.model && models.length > 0) settings.model = models[0];
  } catch (e) {
    checkOk.value = false;
    checkResult.value = t("connection_failed", { error: (e as Error).message });
  }
}
</script>

<template>
  <div class="panel settings-bar">
    <div class="field-row">
      <div class="field grow2">
        <label>{{ t("openai_base_url_label") }}</label>
        <input v-model="settings.openaiBaseUrl" type="text" class="mono full" />
        <div class="help-text">{{ t("openai_base_url_help") }}</div>
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
      <div v-if="props.showFlask" class="field">
        <label>{{ t("flask_base_url_label") }}</label>
        <input v-model="settings.flaskBaseUrl" type="text" class="mono full" />
        <div class="help-text">{{ t("flask_base_url_help") }}</div>
      </div>
      <div class="field check-field">
        <label>&nbsp;</label>
        <button class="small" @click="checkConnection">{{ t("check_connection_btn") }}</button>
      </div>
    </div>
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

.check-field {
  flex: 0 0 auto !important;
}

.field {
  margin-bottom: 0;
}
</style>
