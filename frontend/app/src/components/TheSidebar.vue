<script setup lang="ts">
/** App sidebar: brand mark, primary navigation, language/theme toggles. */
import AppIcon from "./AppIcon.vue";
import logoMark from "../assets/logo-mark.png";
import { useI18n, type MessageKey } from "../i18n";
import type { IconName } from "../lib/icons";
import { settings } from "../lib/settings";

const { t } = useI18n();

const links: { to: string; key: MessageKey; icon: IconName }[] = [
  { to: "/home", key: "nav_home", icon: "home" },
  { to: "/vectors", key: "nav_workshop", icon: "flask" },
  { to: "/steer", key: "nav_playground", icon: "sliders" },
  { to: "/sae", key: "nav_sae", icon: "search" },
  { to: "/chat", key: "nav_chat", icon: "chat" },
  { to: "/gallery", key: "nav_gallery", icon: "grid" },
];

function toggleLanguage(): void {
  settings.language = settings.language === "en" ? "zh" : "en";
}

function toggleTheme(): void {
  settings.theme = settings.theme === "dark" ? "light" : "dark";
}
</script>

<template>
  <aside class="sidebar">
    <RouterLink to="/home" class="brand">
      <img class="brand-mark" :src="logoMark" alt="" />
      <span class="brand-name">{{ t("app_title") }}</span>
    </RouterLink>

    <nav>
      <RouterLink v-for="link in links" :key="link.to" :to="link.to" class="nav-link">
        <AppIcon :name="link.icon" />
        <span>{{ t(link.key) }}</span>
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
  width: 208px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
  padding: 16px 12px 12px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 6px 16px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}

.brand:hover {
  text-decoration: none;
}

.brand-mark {
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  object-fit: contain;
}

.brand-name {
  font-size: 1.02rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 8px;
  color: var(--text-dim);
  font-weight: 500;
  transition:
    background 0.15s,
    color 0.15s;
}

.nav-link:hover {
  background: var(--bg-hover);
  color: var(--text);
  text-decoration: none;
}

.nav-link.router-link-active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}

.sidebar-footer {
  display: flex;
  gap: 6px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
</style>
