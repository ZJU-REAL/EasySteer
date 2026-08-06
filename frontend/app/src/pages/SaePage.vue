<script setup lang="ts">
/**
 * SAE feature explorer, mirroring the legacy SAE tab on the Flask SAE
 * blueprint: semantic feature search and by-index lookup (Neuronpedia,
 * proxied by the backend), feature inspection, and extracting a feature's
 * decoder row as a steering vector that the playground can pick up.
 */
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import SettingsBar from "../components/SettingsBar.vue";
import { useI18n } from "../i18n";
import * as flask from "../lib/flask";
import { playground, replaceSpec } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import { defaultApplySpec, defaultSteeringSpec } from "../lib/spec";

const router = useRouter();
const { t } = useI18n();

type SearchMode = "query" | "index";
const mode = ref<SearchMode>("query");

const modelId = ref("gemma-2-9b");
const saeId = ref("31-gemmascope-res-16k");
const query = ref("");
const featureIndexText = ref("");

const searching = ref(false);
const searchError = ref("");
const results = ref<flask.SaeSearchResult[]>([]);
const details = ref<flask.SaeFeatureDetails | null>(null);
const detailsLoading = ref(false);

// ---- Extraction to a steering vector ----
const vectorName = ref("");
const vectorScale = ref(500);
const targetLayerText = ref("");
const extracting = ref(false);
const extractError = ref("");
const extracted = ref<flask.SaeExtractedVector | null>(null);

const apiKeyReady = computed(() => settings.neuronpediaApiKey.trim() !== "");

/** Best-effort layer guess from SAE ids like "31-gemmascope-res-16k". */
function guessLayer(): number | null {
  const match = saeId.value.match(/^(\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

async function search(): Promise<void> {
  searchError.value = "";
  details.value = null;
  searching.value = true;
  try {
    const resp = await flask.searchSaeFeatures({
      model_id: modelId.value,
      sae_id: saeId.value,
      query: query.value,
      api_key: settings.neuronpediaApiKey,
    });
    results.value = resp.results;
  } catch (e) {
    results.value = [];
    searchError.value = t("sae_error", { error: (e as Error).message });
  } finally {
    searching.value = false;
  }
}

async function lookupFeature(index: number): Promise<void> {
  searchError.value = "";
  detailsLoading.value = true;
  try {
    const resp = await flask.getSaeFeature(
      modelId.value,
      saeId.value,
      index,
      settings.neuronpediaApiKey,
    );
    details.value = resp.feature;
    if (!vectorName.value) vectorName.value = `sae-feature-${index}`;
    extracted.value = null;
  } catch (e) {
    details.value = null;
    searchError.value = t("sae_error", { error: (e as Error).message });
  } finally {
    detailsLoading.value = false;
  }
}

function submit(): void {
  if (mode.value === "query") {
    void search();
  } else {
    const index = parseInt(featureIndexText.value, 10);
    if (Number.isInteger(index)) void lookupFeature(index);
  }
}

const selectedIndex = computed(() => details.value?.basic_info.index ?? null);

async function extractVector(): Promise<void> {
  if (selectedIndex.value === null) return;
  extracting.value = true;
  extractError.value = "";
  try {
    const resp = await flask.extractSaeVector({
      feature_index: selectedIndex.value,
      vector_name: vectorName.value,
      scale: vectorScale.value,
    });
    if (!resp.success || !resp.vector) {
      throw new Error(resp.error ?? "extraction failed");
    }
    extracted.value = resp.vector;
  } catch (e) {
    extractError.value = t("sae_error", { error: (e as Error).message });
  } finally {
    extracting.value = false;
  }
}

/**
 * Seed the playground with a spec steering along the extracted decoder
 * row. The .pt file is loaded server-side via the payload adapter, so
 * the spec carries an inline-payload placeholder (same convention as
 * the gallery's SAE demo).
 */
function useInPlayground(): void {
  if (!extracted.value) return;
  const layer = targetLayerText.value.trim() !== ""
    ? parseInt(targetLayerText.value, 10)
    : guessLayer();
  const spec = defaultSteeringSpec();
  spec.vectors[0].data = {
    __inline_payload__: `vec.from_pt_direction(${JSON.stringify(extracted.value.file_path)}, layers=[${layer ?? 0}])`,
  };
  spec.vectors[0].scale = vectorScale.value;
  if (layer !== null && Number.isInteger(layer)) spec.vectors[0].layers = [layer];
  spec.vectors[0].name = extracted.value.name;
  spec.vectors[0].apply = { ...defaultApplySpec(), phases: ["prompt"], positions: [-1] };
  replaceSpec(spec);
  playground.presetId = null;
  playground.presetModel = "";
  router.push("/playground");
}

function similarity(result: flask.SaeSearchResult): string {
  return result.cosine_similarity === null ? "-" : result.cosine_similarity.toFixed(3);
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>{{ t("sae_title") }}</h1>
    </div>
    <p class="page-intro">{{ t("sae_intro") }}</p>

    <SettingsBar :show-flask="true" />

    <div class="sae-grid">
      <!-- Left: search -->
      <div class="panel search-panel">
        <div class="field-row">
          <div class="field">
            <label>{{ t("sae_model_id_label") }}</label>
            <input v-model="modelId" type="text" class="mono full" />
          </div>
          <div class="field">
            <label>{{ t("sae_id_label") }}</label>
            <input v-model="saeId" type="text" class="mono full" />
          </div>
        </div>
        <div class="field">
          <label>{{ t("sae_api_key_label") }}</label>
          <input v-model="settings.neuronpediaApiKey" type="password" class="mono full" />
          <div class="help-text">{{ t("sae_api_key_help") }}</div>
        </div>

        <div class="mode-tabs">
          <button :class="{ primary: mode === 'query' }" class="small" @click="mode = 'query'">
            {{ t("sae_search_by_query") }}
          </button>
          <button :class="{ primary: mode === 'index' }" class="small" @click="mode = 'index'">
            {{ t("sae_search_by_index") }}
          </button>
        </div>

        <div v-if="mode === 'query'" class="field">
          <label>{{ t("sae_query_label") }}</label>
          <input
            v-model="query"
            type="text"
            class="full"
            :placeholder="t('sae_query_placeholder')"
            @keydown.enter="submit"
          />
        </div>
        <div v-else class="field">
          <label>{{ t("sae_feature_index_label") }}</label>
          <input
            v-model="featureIndexText"
            type="number"
            class="mono full"
            :placeholder="t('sae_feature_index_placeholder')"
            @keydown.enter="submit"
          />
        </div>

        <button
          class="primary"
          :disabled="searching || !apiKeyReady || (mode === 'query' ? !query.trim() : !featureIndexText)"
          @click="submit"
        >
          {{ mode === "query" ? t("sae_search_btn") : t("sae_lookup_btn") }}
        </button>
        <div v-if="searchError" class="help-text text-err">{{ searchError }}</div>

        <template v-if="mode === 'query' && (searching || results.length > 0)">
          <h3 class="results-title">{{ t("sae_results_title") }}</h3>
          <p v-if="searching" class="dim">{{ t("sae_searching") }}</p>
          <div v-else class="results-scroll">
            <table class="results-table">
              <thead>
                <tr>
                  <th>{{ t("sae_feature_id_column") }}</th>
                  <th>{{ t("sae_description_column") }}</th>
                  <th>{{ t("sae_similarity_column") }}</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="result in results" :key="String(result.index)">
                  <td class="mono">{{ result.index }}</td>
                  <td>{{ result.description ?? "-" }}</td>
                  <td class="mono">{{ similarity(result) }}</td>
                  <td>
                    <button class="small" @click="lookupFeature(Number(result.index))">
                      {{ t("sae_details_btn") }}
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-if="!searching && results.length === 0" class="dim">{{ t("sae_no_results") }}</p>
        </template>
      </div>

      <!-- Right: feature details + extraction -->
      <div class="panel detail-panel">
        <h2>{{ t("sae_feature_details") }}</h2>
        <p v-if="detailsLoading" class="dim">{{ t("sae_searching") }}</p>
        <template v-else-if="details">
          <div class="detail-meta">
            <span class="badge mono">{{ details.basic_info.modelId }}</span>
            <span class="badge mono">{{ t("sae_layer_label") }} {{ details.basic_info.layer }}</span>
            <span class="badge mono">#{{ details.basic_info.index }}</span>
            <span v-if="details.sparsity !== null" class="badge mono">
              {{ t("sae_sparsity_label") }} {{ details.sparsity.toFixed(5) }}
            </span>
          </div>
          <div v-if="details.explanation" class="field">
            <label>{{ t("sae_explanation_label") }}</label>
            <p class="explanation">{{ details.explanation }}</p>
          </div>
          <div class="token-columns">
            <div v-if="details.top_activating_tokens.length > 0">
              <label>{{ t("sae_top_activating_tokens") }}</label>
              <ul class="token-list">
                <li v-for="tok in details.top_activating_tokens" :key="tok.token" class="mono">
                  {{ tok.token }} <span class="dim">{{ tok.activation_value.toFixed(2) }}</span>
                </li>
              </ul>
            </div>
            <div v-if="details.top_inhibiting_tokens.length > 0">
              <label>{{ t("sae_top_inhibiting_tokens") }}</label>
              <ul class="token-list">
                <li v-for="tok in details.top_inhibiting_tokens" :key="tok.token" class="mono">
                  {{ tok.token }} <span class="dim">{{ tok.activation_value.toFixed(2) }}</span>
                </li>
              </ul>
            </div>
          </div>
          <div v-if="details.activation_example" class="field">
            <label>{{ t("sae_activation_example") }}</label>
            <div class="activation-example">
              <span class="mono">{{ details.activation_example.context }}</span>
              <div class="help-text">
                {{ t("sae_trigger_token_label") }}:
                <span class="mono">{{ details.activation_example.trigger_token }}</span>
                · {{ t("sae_max_value_label") }}:
                {{ details.activation_example.max_value.toFixed(2) }}
              </div>
            </div>
          </div>

          <hr class="divider" />
          <h3>{{ t("sae_extract_title") }}</h3>
          <div class="help-text">{{ t("sae_extract_help") }}</div>
          <div class="field-row extract-row">
            <div class="field">
              <label>{{ t("sae_vector_name_label") }}</label>
              <input v-model="vectorName" type="text" class="full" />
            </div>
            <div class="field">
              <label>{{ t("scale_label") }}</label>
              <input v-model.number="vectorScale" type="number" step="1" class="mono full" />
            </div>
            <div class="field">
              <label>{{ t("sae_layer_input_label") }}</label>
              <input
                v-model="targetLayerText"
                type="number"
                class="mono full"
                :placeholder="String(guessLayer() ?? '')"
              />
              <div class="help-text">{{ t("sae_layer_input_help") }}</div>
            </div>
          </div>
          <div class="extract-actions">
            <button class="primary" :disabled="extracting || !vectorName" @click="extractVector">
              {{ extracting ? t("sae_extracting") : t("sae_extract_btn") }}
            </button>
            <button v-if="extracted" class="small" @click="useInPlayground">
              {{ t("use_in_playground_btn") }}
            </button>
          </div>
          <div v-if="extracted" class="help-text text-ok">
            {{ t("sae_extract_done", { path: extracted.file_path }) }}
          </div>
          <div v-if="extractError" class="help-text text-err">{{ extractError }}</div>
        </template>
        <p v-else class="dim">{{ t("sae_no_results") }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sae-grid {
  display: grid;
  grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
  gap: 14px;
  align-items: start;
}

@media (max-width: 1100px) {
  .sae-grid {
    grid-template-columns: 1fr;
  }
}

.mode-tabs {
  display: flex;
  gap: 6px;
  margin: 6px 0 10px;
}

.results-title {
  margin-top: 12px;
}

.results-scroll {
  max-height: 380px;
  overflow-y: auto;
}

.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.results-table th {
  text-align: left;
  color: var(--text-dim);
  font-weight: 500;
  padding: 4px 6px;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  background: var(--bg-panel);
}

.results-table td {
  padding: 5px 6px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}

.detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}

.explanation {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.5;
}

.token-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin: 8px 0;
}

.token-list {
  margin: 2px 0 0;
  padding-left: 16px;
  font-size: 12px;
}

.activation-example {
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 8px;
  font-size: 12px;
}

.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 12px 0;
}

.extract-row {
  margin-top: 6px;
}

.extract-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}
</style>
