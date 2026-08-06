import { createRouter, createWebHashHistory } from "vue-router";
import GalleryPage from "./pages/GalleryPage.vue";
import PlaygroundPage from "./pages/PlaygroundPage.vue";
import WorkshopPage from "./pages/WorkshopPage.vue";

export const router = createRouter({
  // Hash history keeps the built app servable from any static path
  // (including the Flask static dir) without server-side rewrites.
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/gallery" },
    { path: "/gallery", name: "gallery", component: GalleryPage },
    { path: "/playground", name: "playground", component: PlaygroundPage },
    { path: "/workshop", name: "workshop", component: WorkshopPage },
  ],
});
