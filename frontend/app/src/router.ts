import { createRouter, createWebHashHistory } from "vue-router";
import ChatPage from "./pages/ChatPage.vue";
import GalleryPage from "./pages/GalleryPage.vue";
import HomePage from "./pages/HomePage.vue";
import PlaygroundPage from "./pages/PlaygroundPage.vue";
import SaePage from "./pages/SaePage.vue";
import WorkshopPage from "./pages/WorkshopPage.vue";

export const router = createRouter({
  // Hash history keeps the built app servable from any static path
  // without server-side rewrites.
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/home" },
    { path: "/home", name: "home", component: HomePage },
    { path: "/chat", name: "chat", component: ChatPage },
    { path: "/steer", name: "steer", component: PlaygroundPage },
    { path: "/gallery", name: "gallery", component: GalleryPage },
    { path: "/vectors", name: "vectors", component: WorkshopPage },
    { path: "/sae", name: "sae", component: SaePage },
  ],
});
