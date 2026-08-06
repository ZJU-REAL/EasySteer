<script setup lang="ts">
/**
 * Demo gallery: slim cards (method, tagline, chips, paper link); the full
 * description, prompt and spec JSON live in an expandable detail opened
 * before "open in playground".
 */
import { ref } from "vue";
import { useRouter } from "vue-router";
import { galleryEntries, type GalleryEntry } from "../data/gallery";
import { useI18n } from "../i18n";
import { loadGalleryEntry } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import { formatIntList } from "../lib/spec";

const router = useRouter();
const { t } = useI18n();

const expandedId = ref<string | null>(null);

interface SpecVectorJson {
  algorithm?: string;
  layers?: number[];
  apply?: { phases?: string[] };
}

function vectors(entry: GalleryEntry): SpecVectorJson[] {
  return entry.spec.vectors as SpecVectorJson[];
}

/** Card chips: algorithm, layer range, phases (plus vector count). */
function chips(entry: GalleryEntry): string[] {
  const vs = vectors(entry);
  const out = [...new Set(vs.map((v) => v.algorithm ?? "direct"))];
  const layerSets = [...new Set(vs.map((v) => formatIntList(v.layers ?? null)))].filter(
    (s) => s !== "",
  );
  if (layerSets.length === 1) {
    out.push(t("gallery_layers_chip", { layers: layerSets[0] }));
  }
  const phases = [...new Set(vs.flatMap((v) => v.apply?.phases ?? []))];
  out.push(phases.join(" + "));
  if (vs.length > 1) out.push(t("gallery_vectors_chip", { n: vs.length }));
  return out;
}

function toggle(entry: GalleryEntry): void {
  expandedId.value = expandedId.value === entry.id ? null : entry.id;
}

function openEntry(entry: GalleryEntry): void {
  loadGalleryEntry(entry);
  router.push("/playground");
}

function specJson(entry: GalleryEntry): string {
  return JSON.stringify(entry.spec, null, 2);
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>{{ t("gallery_title") }}</h1>
    </div>
    <p class="page-intro">{{ t("gallery_intro") }}</p>

    <div class="card-grid">
      <div
        v-for="entry in galleryEntries"
        :key="entry.id"
        class="card panel"
        :class="{ expanded: expandedId === entry.id }"
      >
        <div class="card-main" @click="toggle(entry)">
          <div class="card-header">
            <h2>{{ entry.method }}</h2>
            <span class="expand-hint dim">{{ expandedId === entry.id ? "▾" : "▸" }}</span>
          </div>
          <p class="card-tagline">{{ entry.tagline[settings.language] }}</p>
          <div class="chip-row">
            <span v-for="chip in chips(entry)" :key="chip" class="badge mono">{{ chip }}</span>
          </div>
          <a class="paper-link" :href="entry.paper.url" target="_blank" rel="noopener" @click.stop>
            {{ t("gallery_paper") }} ↗
          </a>
        </div>

        <div v-if="expandedId === entry.id" class="card-detail">
          <div class="detail-model mono dim">{{ entry.model }}</div>
          <p class="detail-description">{{ entry.description[settings.language] }}</p>
          <p v-if="entry.note" class="detail-note dim">{{ entry.note[settings.language] }}</p>
          <div class="detail-label">{{ t("gallery_prompt") }}</div>
          <div class="detail-prompt mono">{{ entry.prompt }}</div>
          <div class="detail-label">{{ t("gallery_spec_preview") }}</div>
          <pre class="code-block detail-spec">{{ specJson(entry) }}</pre>
          <button class="primary open-btn" @click="openEntry(entry)">
            {{ t("open_in_playground") }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 12px;
  align-items: start;
}

.card {
  padding: 0;
  overflow: hidden;
  transition: border-color 0.15s;
}

.card:hover,
.card.expanded {
  border-color: var(--accent);
}

.card-main {
  padding: 12px 14px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-header h2 {
  margin: 0;
}

.card-tagline {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.45;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.paper-link {
  font-size: 11.5px;
}

.card-detail {
  border-top: 1px solid var(--border);
  background: var(--bg-inset);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.detail-model {
  font-size: 11.5px;
}

.detail-description {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.55;
}

.detail-note {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.45;
}

.detail-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
}

.detail-prompt {
  font-size: 11.5px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 8px;
  max-height: 80px;
  overflow-y: auto;
}

.detail-spec {
  max-height: 220px;
  overflow: auto;
  font-size: 11px;
}

.open-btn {
  align-self: flex-start;
}
</style>
