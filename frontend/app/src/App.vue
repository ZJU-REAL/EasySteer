<script setup lang="ts">
import TheSidebar from "./components/TheSidebar.vue";
</script>

<template>
  <div class="layout">
    <TheSidebar />
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

.content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 28px 40px;
  min-width: 0;
}
</style>
