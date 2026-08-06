<script setup lang="ts">
import { useI18n } from "./i18n";
import { settings } from "./lib/settings";

const { t } = useI18n();

function toggleLanguage(): void {
  settings.language = settings.language === "en" ? "zh" : "en";
}

function toggleTheme(): void {
  settings.theme = settings.theme === "dark" ? "light" : "dark";
}
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <span class="brand-name">{{ t("app_title") }}</span>
        <span class="brand-sub">{{ t("app_subtitle") }}</span>
      </div>
      <nav>
        <RouterLink to="/gallery" class="nav-link">{{ t("nav_gallery") }}</RouterLink>
        <RouterLink to="/playground" class="nav-link">{{ t("nav_playground") }}</RouterLink>
        <RouterLink to="/workshop" class="nav-link">{{ t("nav_workshop") }}</RouterLink>
      </nav>
      <div class="sidebar-footer">
        <button class="small" @click="toggleLanguage">{{ t("language_toggle") }}</button>
        <button class="small" @click="toggleTheme">
          {{ settings.theme === "dark" ? "☀" : "☾" }}
        </button>
      </div>
    </aside>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100%;
}

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

.content {
  flex: 1;
  overflow-y: auto;
  padding: 18px 22px;
  min-width: 0;
}
</style>
