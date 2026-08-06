<script setup lang="ts">
/** Landing page: hero, quick-start links, featured demo strip. */
import { useRouter } from "vue-router";
import AppIcon from "../components/AppIcon.vue";
import { galleryEntries, getGalleryEntry, modelShortName } from "../data/gallery";
import { useI18n, type MessageKey } from "../i18n";
import type { IconName } from "../lib/icons";
import { loadGalleryEntry } from "../lib/playgroundStore";
import { settings } from "../lib/settings";

const router = useRouter();
const { t } = useI18n();

const quickstarts: { to: string; title: MessageKey; text: MessageKey; icon: IconName }[] = [
  { to: "/vectors", title: "home_workshop_title", text: "home_workshop_text", icon: "flask" },
  {
    to: "/steer",
    title: "home_playground_title",
    text: "home_playground_text",
    icon: "sliders",
  },
  { to: "/sae", title: "home_sae_title", text: "home_sae_text", icon: "search" },
  { to: "/chat", title: "home_chat_title", text: "home_chat_text", icon: "chat" },
  { to: "/gallery", title: "home_gallery_title", text: "home_gallery_text", icon: "grid" },
];

const resources: { href: string; title: MessageKey; sub: MessageKey; icon: IconName }[] = [
  {
    href: "https://arxiv.org/abs/2509.25175",
    title: "home_link_paper",
    sub: "home_link_paper_sub",
    icon: "inbox",
  },
  {
    href: "https://github.com/ZJU-REAL/EasySteer",
    title: "home_link_github",
    sub: "home_link_github_sub",
    icon: "grid",
  },
  {
    href: "https://zju-real.github.io/EasySteer/latest/",
    title: "home_link_docs",
    sub: "home_link_docs_sub",
    icon: "search",
  },
  {
    href: "https://huggingface.co/spaces/zjuxhl/EasySteer",
    title: "home_link_demo",
    sub: "home_link_demo_sub",
    icon: "spark",
  },
];

const featured = ["cast", "refusal_direction", "controlingthinkingspeed", "loreft"]
  .map((id) => getGalleryEntry(id))
  .filter((e) => e !== undefined);

function openDemo(id: string): void {
  const entry = galleryEntries.find((e) => e.id === id);
  if (!entry) return;
  loadGalleryEntry(entry);
  router.push("/steer");
}
</script>

<template>
  <div class="page home">
    <section class="hero panel">
      <h1 class="hero-paper">{{ t("app_title") }}: {{ t("home_paper_title") }}</h1>

      <nav class="resources" :aria-label="t('home_resources_title')">
        <a
          v-for="link in resources"
          :key="link.href"
          class="resource"
          :href="link.href"
          target="_blank"
          rel="noopener"
        >
          <span class="resource-icon"><AppIcon :name="link.icon" :size="15" /></span>
          <span class="resource-text">
            <span class="resource-title">{{ t(link.title) }}</span>
            <span class="resource-sub dim">{{ t(link.sub) }}</span>
          </span>
          <AppIcon name="external" :size="12" class="resource-arrow" />
        </a>
      </nav>
    </section>

    <section>
      <h2 class="section-title">{{ t("home_quickstart_title") }}</h2>
      <div class="quickstart-grid">
        <RouterLink v-for="q in quickstarts" :key="q.to" :to="q.to" class="quickstart panel">
          <span class="quickstart-icon"><AppIcon :name="q.icon" :size="18" /></span>
          <h3>{{ t(q.title) }}</h3>
          <p>{{ t(q.text) }}</p>
        </RouterLink>
      </div>
    </section>

    <section>
      <div class="featured-header">
        <h2 class="section-title">{{ t("home_featured_title") }}</h2>
        <RouterLink to="/gallery">{{ t("home_featured_more") }} →</RouterLink>
      </div>
      <div class="featured-strip">
        <button
          v-for="entry in featured"
          :key="entry!.id"
          class="featured-card panel"
          :style="{ '--tone': `var(--tone-${entry!.category})` }"
          @click="openDemo(entry!.id)"
        >
          <span class="featured-top">
            <span class="featured-method">{{ entry!.method }}</span>
            <AppIcon name="spark" :size="14" />
          </span>
          <span class="featured-tagline dim">{{ entry!.tagline[settings.language] }}</span>
          <span class="featured-model mono dim">{{ modelShortName(entry!.model) }}</span>
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home {
  max-width: 1120px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.hero {
  position: relative;
  overflow: hidden;
  padding: 32px 28px 26px;
  border-radius: var(--radius-lg);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  background:
    radial-gradient(620px 260px at 50% -40%, var(--accent-soft), transparent 70%),
    var(--bg-panel);
}

/* Decorative dot grid, fading toward the centred text. */
.hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image: radial-gradient(currentColor 1px, transparent 1px);
  background-size: 16px 16px;
  color: var(--accent);
  opacity: 0.1;
  mask-image: radial-gradient(closest-side at 50% 40%, transparent 40%, #000);
  pointer-events: none;
}

.hero > * {
  position: relative;
  z-index: 1;
}

.hero-paper {
  font-size: 1.45rem;
  font-weight: 650;
  line-height: 1.35;
  max-width: 44ch;
  margin: 0 0 20px;
}

.resources {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
}

.resource {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 13px 8px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-panel);
  color: var(--text);
  text-align: left;
  transition:
    border-color 0.15s,
    box-shadow 0.15s,
    transform 0.15s;
}

.resource:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
  text-decoration: none;
}

.resource-icon {
  display: grid;
  place-items: center;
  width: 26px;
  height: 26px;
  flex-shrink: 0;
  border-radius: 50%;
  color: var(--accent);
  background: var(--accent-soft);
}

.resource-text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.resource-title {
  font-size: 12.5px;
  font-weight: 600;
}

.resource-sub {
  font-size: 10.5px;
}

.resource-arrow {
  color: var(--text-dim);
  flex-shrink: 0;
}

.quickstart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(178px, 1fr));
  gap: 12px;
}

.quickstart {
  display: block;
  color: var(--text);
  transition:
    transform 0.15s,
    box-shadow 0.15s,
    border-color 0.15s;
}

.quickstart:hover {
  transform: translateY(-2px);
  border-color: var(--border-strong);
  box-shadow: var(--shadow);
  text-decoration: none;
}

.quickstart-icon {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  margin-bottom: 9px;
  border-radius: 8px;
  color: var(--accent);
  background: var(--accent-soft);
}

.quickstart h3 {
  margin-bottom: 4px;
}

.quickstart p {
  margin: 0;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.5;
}

.featured-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.featured-header .section-title {
  margin-bottom: 10px;
}

.featured-strip {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
  gap: 12px;
}

.featured-card {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 6px;
  text-align: left;
  cursor: pointer;
  border-top: 3px solid var(--tone);
  transition:
    transform 0.15s,
    box-shadow 0.15s;
}

.featured-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
  background: var(--bg-panel);
}

.featured-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--tone);
}

.featured-method {
  font-weight: 650;
  color: var(--text);
}

.featured-tagline {
  font-size: 11.5px;
  line-height: 1.45;
  flex: 1;
}

.featured-model {
  font-size: 10.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
