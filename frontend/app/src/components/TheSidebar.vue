<script setup lang="ts">
/** App sidebar: brand, primary navigation, language/theme toggles. */
import { useI18n } from "../i18n";
import { settings } from "../lib/settings";

const { t } = useI18n();

const links = [
  { to: "/home", key: "nav_home" },
  { to: "/chat", key: "nav_chat" },
  { to: "/playground", key: "nav_playground" },
  { to: "/gallery", key: "nav_gallery" },
  { to: "/workshop", key: "nav_workshop" },
  { to: "/sae", key: "nav_sae" },
] as const;

function toggleLanguage(): void {
  settings.language = settings.language === "en" ? "zh" : "en";
}

function toggleTheme(): void {
  settings.theme = settings.theme === "dark" ? "light" : "dark";
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <span class="brand-name">{{ t("app_title") }}</span>
      <span class="brand-sub">{{ t("app_subtitle") }}</span>
    </div>
    <nav>
      <RouterLink v-for="link in links" :key="link.to" :to="link.to" class="nav-link">
        {{ t(link.key) }}
      </RouterLink>
    </nav>
    <div class="sidebar-footer">
      <button class="small" @click="toggleLanguage">{{ t("language_toggle") }}</button>
      <button class="small" :title="t('theme_toggle')" @click="toggleTheme">
        {{ settings.theme === "dark" ? "☀" : "☾" }}
      </button>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 190px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
  padding: 14px 10px;
}

.brand {
  padding: 0 8px 14px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
}

.brand-name {
  display: block;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.brand-sub {
  display: block;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 2px;
}

nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.nav-link {
  padding: 7px 10px;
  border-radius: 6px;
  color: var(--text);
  font-weight: 500;
}

.nav-link:hover {
  background: var(--bg-hover);
  text-decoration: none;
}

.nav-link.router-link-active {
  background: var(--accent-soft);
  color: var(--accent);
}

.sidebar-footer {
  display: flex;
  gap: 6px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
</style>
