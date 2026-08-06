<script setup lang="ts">
/**
 * SAE feature explorer, mirroring the legacy SAE tab on the Flask SAE
 * blueprint: semantic feature search and by-index lookup (Neuronpedia,
 * proxied by the backend), feature inspection, and extracting a feature's
 * decoder row as a steering vector that the playground can pick up.
 */
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import AppIcon from "../components/AppIcon.vue";
import { useI18n } from "../i18n";
import * as flask from "../lib/flask";
import { loadCustomSpec } from "../lib/playgroundStore";
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
  spec.vectors[0].apply = { ...defaultApplySpec(), phases: ["prompt"], prompt_positions: [-1] };
  loadCustomSpec(spec);
  router.push("/steer");
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

    <!-- Search: mode tabs on top of one compact form, as in the old UI -->
    <div class="panel search-panel">
      <div class="panel-header">
        <h2>{{ t("sae_search_title") }}</h2>
        <span class="spacer"></span>
        <div class="tab-bar">
          <button class="tab" :class="{ active: mode === 'query' }" @click="mode = 'query'">
            {{ t("sae_search_by_query") }}
          </button>
          <button class="tab" :class="{ active: mode === 'index' }" @click="mode = 'index'">
            {{ t("sae_search_by_index") }}
          </button>
        </div>
      </div>

      <div class="field-row">
        <div class="field">
          <label>{{ t("sae_model_id_label") }}</label>
          <input v-model="modelId" type="text" class="mono full" />
          <div class="help-text">{{ t("sae_model_id_help") }}</div>
        </div>
        <div class="field">
          <label>{{ t("sae_id_label") }}</label>
          <input v-model="saeId" type="text" class="mono full" />
          <div class="help-text">{{ t("sae_id_help") }}</div>
        </div>
        <div class="field">
          <label>{{ t("sae_api_key_label") }}</label>
          <input v-model="settings.neuronpediaApiKey" type="password" class="mono full" />
          <div class="help-text">{{ t("sae_api_key_help") }}</div>
        </div>
      </div>

      <div class="field-row query-row">
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
        <div class="field submit-field">
          <button
            class="primary"
            :disabled="
              searching || !apiKeyReady || (mode === 'query' ? !query.trim() : !featureIndexText)
            "
            @click="submit"
          >
            {{ mode === "query" ? t("sae_search_btn") : t("sae_lookup_btn") }}
          </button>
        </div>
      </div>
      <div v-if="searchError" class="help-text text-err">{{ searchError }}</div>
    </div>

    <!-- Results -->
    <div v-if="mode === 'query' && (searching || results.length > 0)" class="panel results-panel">
      <div class="panel-header">
        <h2>{{ t("sae_results_title") }}</h2>
      </div>
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
    </div>

    <!-- Feature details + extraction -->
    <div class="panel detail-panel">
      <div class="panel-header">
        <h2>{{ t("sae_feature_details") }}</h2>
        <span v-if="details" class="badge mono">#{{ details.basic_info.index }}</span>
      </div>
      <p v-if="detailsLoading" class="dim">{{ t("sae_searching") }}</p>
      <template v-else-if="details">
        <div class="detail-meta">
          <span class="badge mono">{{ details.basic_info.modelId }}</span>
          <span class="badge mono">{{ t("sae_layer_label") }} {{ details.basic_info.layer }}</span>
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
          <div v-if="details.activation_example">
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
        </div>

        <hr class="divider" />
        <h3>{{ t("sae_extract_title") }}</h3>
        <div class="field-row extract-row">
          <div class="field">
            <label>{{ t("sae_vector_name_label") }}</label>
            <input v-model="vectorName" type="text" class="full" />
            <div class="help-text">{{ t("sae_vector_name_help") }}</div>
          </div>
          <div class="field">
            <label>{{ t("scale_label") }}</label>
            <input v-model.number="vectorScale" type="number" step="1" class="mono full" />
            <div class="help-text">{{ t("sae_scale_help") }}</div>
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
          <button v-if="extracted" @click="useInPlayground">
            {{ t("use_in_playground_btn") }}
          </button>
          <span class="help-text extract-note">{{ t("sae_extract_help") }}</span>
        </div>
        <div v-if="extracted" class="help-text text-ok">
          {{ t("sae_extract_done", { path: extracted.file_path }) }}
        </div>
        <div v-if="extractError" class="help-text text-err">{{ extractError }}</div>
      </template>
      <div v-else class="empty-state">
        <AppIcon name="search" :size="30" />
        <p>{{ t("sae_details_placeholder") }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.search-panel,
.results-panel {
  margin-bottom: 14px;
}

.search-panel .field-row {
  align-items: flex-end;
}

.query-row .field {
  margin-bottom: 0;
}

.submit-field {
  flex: 0 0 auto !important;
}

.results-scroll {
  max-height: 320px;
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
  padding: 6px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}

/* Fixed index/score/action columns so the description takes the slack
   and the numbers line up under their header. */
.results-table th:first-child,
.results-table td:first-child {
  width: 62px;
}

.results-table th:nth-child(3),
.results-table td:nth-child(3) {
  width: 86px;
  text-align: right;
}

.results-table th:last-child,
.results-table td:last-child {
  width: 78px;
  text-align: right;
}

.detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.explanation {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.5;
}

.token-columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  margin: 10px 0;
  align-items: start;
}

.token-list {
  margin: 2px 0 0;
  padding-left: 16px;
  font-size: 12px;
}

.activation-example {
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 9px;
  font-size: 12px;
}

.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 14px 0 12px;
}

.extract-row {
  margin-top: 8px;
}

.extract-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 4px;
}

.extract-note {
  margin-top: 0;
}
</style>
