/** Inline icon paths on a 24x24 stroke grid, so the app ships no icon font. */

export type IconName =
  | "home"
  | "chat"
  | "sliders"
  | "grid"
  | "flask"
  | "search"
  | "spark"
  | "inbox"
  | "external";

export const ICON_PATHS: Record<IconName, string[]> = {
  home: ["M4 11.2 12 4.2l8 7M6.4 9.8V19.8h11.2V9.8"],
  chat: [
    "M20 11.6c0 3.8-3.6 6.9-8 6.9-.9 0-1.8-.1-2.6-.4L5 19.5l1.3-3.2C5.1 15.1 4 13.5 4 11.6c0-3.8 3.6-6.9 8-6.9s8 3.1 8 6.9z",
  ],
  sliders: [
    "M4 7.5h6M14 7.5h6M4 16.5h10M18 16.5h2",
    "M12 7.5a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM18 16.5a2 2 0 1 1-4 0 2 2 0 0 1 4 0z",
  ],
  grid: ["M4.5 4.5h6v6h-6zM13.5 4.5h6v6h-6zM4.5 13.5h6v6h-6zM13.5 13.5h6v6h-6z"],
  flask: ["M9.5 3.5h5M10.5 3.5v6L5.9 17.2A2 2 0 0 0 7.6 20.3h8.8a2 2 0 0 0 1.7-3.1L13.5 9.5v-6"],
  search: ["M17 11a6 6 0 1 1-12 0 6 6 0 0 1 12 0zM15.4 15.4 20 20"],
  spark: ["M12 4.5 13.7 10 19 11.8 13.7 13.6 12 19l-1.7-5.4L5 11.8 10.3 10z"],
  inbox: ["M4 13.5h4l1.5 2.5h5l1.5-2.5h4M4 13.5 6.5 5.5h11L20 13.5v5H4z"],
  external: ["M14 4.5h5.5V10M19.5 4.5 11 13M17 14v5.5H4.5V7H10"],
};
