import { afterEach, describe, expect, it, vi } from "vitest";

import { setServerSteering, streamChatCompletion } from "./openai";

afterEach(() => vi.unstubAllGlobals());

describe("default steering requests", () => {
  it.each([false, null] as const)("preserves the %s request choice", async (steering) => {
    const fetchMock = vi.fn<typeof fetch>(async () => new Response("data: [DONE]\n\n"));
    vi.stubGlobal("fetch", fetchMock);
    await streamChatCompletion({
      baseUrl: "http://server/v1", model: "test", messages: [],
      steering, onToken: () => {},
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
    if (steering === false) expect(body.steering).toBe(false);
    else expect(body).not.toHaveProperty("steering");
  });

  it("clears the default with an explicit null spec", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => Response.json({ active: false }));
    vi.stubGlobal("fetch", fetchMock);
    await setServerSteering("http://server/v1", null);
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({ spec: null });
  });
});
