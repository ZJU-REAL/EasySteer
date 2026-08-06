<script setup lang="ts">
/** Landing page: one card per replication notebook. */
import { useRouter } from "vue-router";
import { galleryEntries, type GalleryEntry } from "../data/gallery";
import { useI18n } from "../i18n";
import { loadGalleryEntry } from "../lib/playgroundStore";
import { settings } from "../lib/settings";

const router = useRouter();
const { t } = useI18n();

function algorithms(entry: GalleryEntry): string[] {
  const vectors = entry.spec.vectors as { algorithm?: string }[];
  return [...new Set(vectors.map((v) => v.algorithm ?? "direct"))];
}

function vectorCount(entry: GalleryEntry): number {
  return (entry.spec.vectors as unknown[]).length;
}

function openEntry(entry: GalleryEntry): void {
  loadGalleryEntry(entry);
  router.push("/playground");
}
</script>

<template>
  <div>
    <h1>{{ t("gallery_title") }}</h1>
    <p class="dim intro">{{ t("gallery_intro") }}</p>
    <div class="card-grid">
      <div v-for="entry in galleryEntries" :key="entry.id" class="card panel" @click="openEntry(entry)">
        <div class="card-header">
          <h2>{{ entry.method }}</h2>
          <span v-for="algo in algorithms(entry)" :key="algo" class="badge accent mono">{{
            algo
          }}</span>
          <span v-if="vectorCount(entry) > 1" class="badge mono"
            >{{ vectorCount(entry) }}x</span
          >
        </div>
        <div class="card-model mono dim">{{ entry.model }}</div>
        <p class="card-description">{{ entry.description[settings.language] }}</p>
        <p v-if="entry.note" class="card-note dim">{{ entry.note[settings.language] }}</p>
        <div class="card-prompt mono">"{{ entry.prompt }}"</div>
        <div class="card-footer">
          <a :href="entry.paper.url" target="_blank" rel="noopener" @click.stop>
            {{ t("gallery_paper") }}: {{ entry.paper.title }}
          </a>
          <button class="small primary" @click.stop="openEntry(entry)">
            {{ t("open_in_playground") }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.intro {
  margin-top: 0;
  max-width: 70ch;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 12px;
}

.card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  cursor: pointer;
  transition: border-color 0.15s;
}

.card:hover {
  border-color: var(--accent);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.card-header h2 {
  margin: 0;
  margin-right: 4px;
}

.card-model {
  font-size: 11.5px;
}

.card-description {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.5;
}

.card-note {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.4;
}

.card-prompt {
  font-size: 11.5px;
  color: var(--text-dim);
  background: var(--bg-inset);
  border-radius: 5px;
  padding: 5px 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.card-footer {
  margin-top: auto;
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 8px;
  padding-top: 4px;
}

.card-footer a {
  font-size: 11.5px;
  flex: 1;
}

.card-footer button {
  flex-shrink: 0;
}
</style>
