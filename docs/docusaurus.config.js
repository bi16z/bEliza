// @ts-check
import { themes as prismThemes } from "prism-react-renderer";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "beliza",
  tagline: "The flexible, scalable AI agent for everyone",
  favicon: "img/favicon.ico",
  url: "https://docs.bi16z.ai",
  baseUrl: "/",
  organizationName: "bi16z",
  projectName: "beliza",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  plugins: [
    // TypeDoc plugin for API documentation
    [
      "docusaurus-plugin-typedoc",
      {
        entryPoints: ["../src/index.ts"],
        tsconfig: "../tsconfig.json",
        out: "./api", // Changed to output directly to api folder
      },
    ],
    // Search functionality
    require.resolve("docusaurus-lunr-search"),
    // Separate API docs plugin instance
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "api",
        path: "api",
        routeBasePath: "api",
        sidebarPath: "./sidebars.api.js",
      },
    ],
  ],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: "./sidebars.js",
          editUrl: "https://github.com/bi16z/bbeliza/tree/main/docs/",
          routeBasePath: "docs",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "beliza",
        logo: {
          alt: "bEliza Logo",
          src: "img/favicon.ico",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "tutorialSidebar",
            position: "left",
            label: "Documentation",
          },
          {
            type: "doc",
            docsPluginId: "api",
            position: "left",
            label: "API",
            docId: "index",
          },
          {
            href: "https://github.com/bi16z/bbeliza",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      // ... rest of themeConfig remains the same
    }),
};

export default config;
