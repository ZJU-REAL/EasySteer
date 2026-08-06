<script setup lang="ts">
/**
 * Demo gallery: uniform cards (category, method, tagline, model, year)
 * filtered by category. Details open in a modal, so opening one card
 * never changes the height of its neighbours.
 */
import { computed, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import AppIcon from "../components/AppIcon.vue";
import ModalDialog from "../components/ModalDialog.vue";
import {
  DEMO_CATEGORIES,
  galleryEntries,
  modelShortName,
  type DemoCategory,
  type GalleryEntry,
} from "../data/gallery";
import { useI18n, type MessageKey } from "../i18n";
import { loadGalleryEntry } from "../lib/playgroundStore";
import { settings } from "../lib/settings";
import { formatIntList } from "../lib/spec";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();

const filter = ref<DemoCategory | "all">("all");

// The open card lives in the URL, so a demo can be linked to directly.
const openEntryId = computed<string | null>({
  get: () => (typeof route.query.demo === "string" ? route.query.demo : null),
  set: (id) => {
    router.replace({ query: id ? { demo: id } : {} });
  },
});

const visible = computed(() =>
  filter.value === "all"
    ? galleryEntries
    : galleryEntries.filter((e) => e.category === filter.value),
);

const openedEntry = computed(() => galleryEntries.find((e) => e.id === openEntryId.value));

function categoryLabel(category: DemoCategory): string {
  return t(`cat_${category}` as MessageKey);
}

function countOf(category: DemoCategory): number {
  return galleryEntries.filter((e) => e.category === category).length;
}

interface SpecVectorJson {
  algorithm?: string;
  layers?: number[];
  scale?: number;
  apply?: { phases?: string[] };
}

/** One line of technical detail for the modal: algorithm, layers, phases. */
function specSummary(entry: GalleryEntry): string {
  const vs = entry.spec.vectors as SpecVectorJson[];
  const algorithms = [...new Set(vs.map((v) => v.algorithm ?? "direct"))].join(", ");
  const layers = [...new Set(vs.map((v) => formatIntList(v.layers ?? null)))]
    .filter((s) => s !== "")
    .join(" / ");
  const phases = [...new Set(vs.flatMap((v) => v.apply?.phases ?? []))].join(" + ");
  const parts = [algorithms];
  if (layers) parts.push(t("gallery_layers_chip", { layers }));
  if (phases) parts.push(phases);
  if (vs.length > 1) parts.push(t("gallery_vectors_chip", { n: vs.length }));
  return parts.join(" · ");
}

function openInPlayground(entry: GalleryEntry): void {
  loadGalleryEntry(entry);
  router.push("/steer");
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>{{ t("gallery_title") }}</h1>
      <span class="badge">{{ t("gallery_count", { n: galleryEntries.length }) }}</span>
    </div>
    <p class="page-intro">{{ t("gallery_intro") }}</p>

    <div class="tab-bar filter-bar">
      <button class="tab" :class="{ active: filter === 'all' }" @click="filter = 'all'">
        {{ t("gallery_all_filter") }}
      </button>
      <button
        v-for="category in DEMO_CATEGORIES"
        :key="category"
        class="tab"
        :class="{ active: filter === category }"
        :style="{ '--tone': `var(--tone-${category})` }"
        @click="filter = category"
      >
        <span class="tone-dot"></span>
        {{ categoryLabel(category) }}
        <span class="dim">{{ countOf(category) }}</span>
      </button>
    </div>

    <div class="card-grid">
      <button
        v-for="entry in visible"
        :key="entry.id"
        class="demo-card panel"
        :style="{ '--tone': `var(--tone-${entry.category})` }"
        @click="openEntryId = entry.id"
      >
        <span class="card-top">
          <span class="chip tone">
            <span class="dot"></span>
            {{ categoryLabel(entry.category) }}
          </span>
          <span class="card-year dim">{{ entry.year }}</span>
        </span>
        <span class="card-method">{{ entry.method }}</span>
        <span class="card-tagline dim">{{ entry.tagline[settings.language] }}</span>
        <span class="card-foot mono dim">{{ modelShortName(entry.model) }}</span>
      </button>
    </div>

    <ModalDialog
      v-if="openedEntry"
      :title="openedEntry.method"
      width="720px"
      @close="openEntryId = null"
    >
      <template #head-actions>
        <span class="chip tone" :style="{ '--tone': `var(--tone-${openedEntry.category})` }">
          <span class="dot"></span>
          {{ categoryLabel(openedEntry.category) }}
        </span>
      </template>

      <p class="detail-description">{{ openedEntry.description[settings.language] }}</p>
      <p v-if="openedEntry.note" class="detail-note dim">
        {{ openedEntry.note[settings.language] }}
      </p>

      <dl class="detail-meta">
        <dt>{{ t("gallery_model") }}</dt>
        <dd class="mono">{{ openedEntry.model }}</dd>
        <dt>{{ t("gallery_how") }}</dt>
        <dd class="mono">{{ specSummary(openedEntry) }}</dd>
        <dt>{{ t("gallery_paper") }}</dt>
        <dd>
          <a :href="openedEntry.paper.url" target="_blank" rel="noopener">
            {{ openedEntry.paper.title }}
            <AppIcon name="external" :size="13" />
          </a>
        </dd>
      </dl>

      <h3 class="detail-label">{{ t("gallery_prompt") }}</h3>
      <pre class="code-block detail-prompt">{{ openedEntry.prompt }}</pre>

      <h3 class="detail-label">{{ t("gallery_spec_preview") }}</h3>
      <pre class="code-block detail-spec">{{ JSON.stringify(openedEntry.spec, null, 2) }}</pre>

      <template #footer>
        <button class="primary" @click="openInPlayground(openedEntry)">
          {{ t("open_in_playground") }}
        </button>
        <span class="spacer"></span>
        <button @click="openEntryId = null">{{ t("close_btn") }}</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.filter-bar {
  margin-bottom: 14px;
}

.tone-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--tone, var(--accent));
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(272px, 1fr));
  gap: 12px;
}

.demo-card {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 7px;
  text-align: left;
  padding: 0 14px 12px;
  cursor: pointer;
  overflow: hidden;
  position: relative;
  transition:
    transform 0.15s,
    box-shadow 0.15s,
    border-color 0.15s;
}

/* Category rail across the top of every card. */
.demo-card::before {
  content: "";
  position: absolute;
  inset: 0 0 auto;
  height: 3px;
  background: var(--tone);
  opacity: 0.85;
}

.demo-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-strong);
  box-shadow: var(--shadow);
  background: var(--bg-panel);
}

.card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 14px;
}

.card-year {
  font-size: 11.5px;
}

.card-method {
  font-size: 1rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}

/* Grows to fill the card, keeping every footer on the same baseline. */
.card-tagline {
  font-size: 12.5px;
  line-height: 1.45;
  flex: 1;
  min-height: 2.9em;
}

.card-foot {
  font-size: 11px;
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-description {
  margin: 0 0 8px;
  line-height: 1.6;
}

.detail-note {
  margin: 0 0 12px;
  font-size: 12px;
  line-height: 1.5;
}

.detail-meta {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 6px 14px;
  margin: 0 0 16px;
  padding: 12px 14px;
  background: var(--bg-inset);
  border-radius: var(--radius-sm);
  font-size: 12px;
}

.detail-meta dt {
  color: var(--text-dim);
  font-weight: 500;
}

.detail-meta dd {
  margin: 0;
  min-width: 0;
  overflow-wrap: anywhere;
}

.detail-meta a {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
}

.detail-label {
  margin: 0 0 6px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
}

.detail-prompt {
  margin-bottom: 16px;
  max-height: 130px;
  overflow: auto;
  white-space: pre-wrap;
}

.detail-spec {
  max-height: 260px;
  overflow: auto;
  font-size: 11.5px;
}
</style>
