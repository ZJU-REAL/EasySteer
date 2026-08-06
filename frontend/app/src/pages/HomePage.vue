<script setup lang="ts">
/** Landing page: short intro, quick-start links, featured demo strip. */
import { useRouter } from "vue-router";
import { galleryEntries, getGalleryEntry } from "../data/gallery";
import { useI18n, type MessageKey } from "../i18n";
import { loadGalleryEntry } from "../lib/playgroundStore";
import { settings } from "../lib/settings";

const router = useRouter();
const { t } = useI18n();

const quickstarts: { to: string; title: MessageKey; text: MessageKey }[] = [
  { to: "/chat", title: "home_chat_title", text: "home_chat_text" },
  { to: "/playground", title: "home_playground_title", text: "home_playground_text" },
  { to: "/gallery", title: "home_gallery_title", text: "home_gallery_text" },
  { to: "/workshop", title: "home_workshop_title", text: "home_workshop_text" },
  { to: "/sae", title: "home_sae_title", text: "home_sae_text" },
];

const featured = ["cast", "refusal_direction", "controlingthinkingspeed", "loreft"]
  .map((id) => getGalleryEntry(id))
  .filter((e) => e !== undefined);

function openDemo(id: string): void {
  const entry = galleryEntries.find((e) => e.id === id);
  if (!entry) return;
  loadGalleryEntry(entry);
  router.push("/playground");
}
</script>

<template>
  <div class="page home">
    <section class="hero">
      <h1>{{ t("app_title") }}</h1>
      <p class="hero-tagline">{{ t("home_tagline") }}</p>
      <p class="hero-intro">{{ t("home_intro") }}</p>
    </section>

    <section>
      <h2>{{ t("home_quickstart_title") }}</h2>
      <div class="quickstart-grid">
        <RouterLink v-for="q in quickstarts" :key="q.to" :to="q.to" class="quickstart panel">
          <h3>{{ t(q.title) }}</h3>
          <p>{{ t(q.text) }}</p>
        </RouterLink>
      </div>
    </section>

    <section>
      <div class="featured-header">
        <h2>{{ t("home_featured_title") }}</h2>
        <RouterLink to="/gallery">{{ t("home_featured_more") }} →</RouterLink>
      </div>
      <div class="featured-strip">
        <button
          v-for="entry in featured"
          :key="entry!.id"
          class="featured-card panel"
          @click="openDemo(entry!.id)"
        >
          <span class="featured-method">{{ entry!.method }}</span>
          <span class="featured-tagline dim">{{ entry!.tagline[settings.language] }}</span>
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home {
  max-width: 980px;
  display: flex;
  flex-direction: column;
  gap: 26px;
}

.hero h1 {
  font-size: 1.7rem;
  margin-bottom: 6px;
}

.hero-tagline {
  font-size: 1.05rem;
  margin: 0 0 10px;
}

.hero-intro {
  margin: 0;
  max-width: 72ch;
  color: var(--text-dim);
  line-height: 1.6;
}

.quickstart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 10px;
}

.quickstart {
  display: block;
  color: var(--text);
  transition: border-color 0.15s;
}

.quickstart:hover {
  border-color: var(--accent);
  text-decoration: none;
}

.quickstart h3 {
  color: var(--accent);
  margin-bottom: 4px;
}

.quickstart p {
  margin: 0;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.45;
}

.featured-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
}

.featured-header h2 {
  margin: 0;
}

.featured-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 10px;
}

.featured-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s;
}

.featured-card:hover {
  border-color: var(--accent);
  background: var(--bg-panel);
}

.featured-method {
  font-weight: 600;
}

.featured-tagline {
  font-size: 11.5px;
  line-height: 1.4;
}
</style>
