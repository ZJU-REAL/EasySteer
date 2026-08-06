import { defineConfig } from "vitest/config";
import type { Plugin } from "vite";
import vue from "@vitejs/plugin-vue";

/**
 * Dev-only mock of the vllm-steer OpenAI-compatible server.
 *
 * Point the UI base URL at `http://localhost:5173/mock/v1` to exercise the
 * playground without a GPU server. Streams a canned SSE completion that
 * echoes the request so steering payloads are visible end to end.
 */
function mockOpenAIServer(): Plugin {
  return {
    name: "easysteer-mock-openai",
    configureServer(server) {
      server.middlewares.use("/mock/v1/chat/completions", (req, res) => {
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
          let parsed: Record<string, unknown> = {};
          try {
            parsed = JSON.parse(body);
          } catch {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "invalid JSON body" }));
            return;
          }
          const steering = parsed["steering"];
          const words = [
            "[mock]",
            "This",
            "is",
            "a",
            "simulated",
            "completion",
            "from",
            "the",
            "dev",
            "mock",
            "server.",
            steering ? "Steering spec received: " + JSON.stringify(steering) : "No steering spec attached.",
          ];
          res.setHeader("Content-Type", "text/event-stream");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Access-Control-Allow-Origin", "*");
          let i = 0;
          const timer = setInterval(() => {
            if (i < words.length) {
              const chunk = {
                id: "mock",
                object: "chat.completion.chunk",
                choices: [{ index: 0, delta: { content: words[i] + " " }, finish_reason: null }],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              i += 1;
            } else {
              res.write("data: [DONE]\n\n");
              clearInterval(timer);
              res.end();
            }
          }, 60);
        });
      });
      server.middlewares.use("/mock/v1/steering", (req, res) => {
        let body = "";
        req.on("data", (chunk) => (body += chunk));
        req.on("end", () => {
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.end(JSON.stringify({ ok: true, echoed: body ? JSON.parse(body) : null }));
        });
      });
      server.middlewares.use("/mock/v1/models", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.end(
          JSON.stringify({ object: "list", data: [{ id: "mock-model", object: "model" }] }),
        );
      });
    },
  };
}

export default defineConfig({
  plugins: [vue(), mockOpenAIServer()],
  server: {
    proxy: {
      // Flask job backend (extraction / training) during development.
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
