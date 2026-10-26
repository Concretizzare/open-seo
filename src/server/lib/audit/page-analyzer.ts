/**
 * HTML page analyzer using htmlparser2's streaming tokenizer.
 *
 * Extracts SEO-relevant data from a page's HTML: title, meta description,
 * headings, images, links, canonical, OG tags, structured data, robots meta,
 * word count, hreflang.
 *
 * Deliberately NOT a DOM parser: the previous cheerio implementation built a
 * full DOM (~5-10x the HTML's size) per page, and with 25 concurrent parses
 * on a 128MB isolate that was the audit engine's dominant OOM cause. The
 * tokenizer keeps only the accumulated text and extracted fields in memory.
 */
import { Parser } from "htmlparser2";
import { z } from "zod";
import { normalizeUrl, isSameOrigin } from "./url-utils";
import type { PageAnalysis, PageLink } from "./types";

const SKIPPED_LINK_PROTOCOLS = /^(javascript:|mailto:|tel:|#)/;
/** Subtrees whose text is not visible content. */
const NON_CONTENT_TAGS = new Set(["script", "style", "noscript", "svg"]);
const HEADING_LEVELS: Record<string, number> = {
  h1: 1,
  h2: 2,
  h3: 3,
  h4: 4,
  h5: 5,
  h6: 6,
};
const MAX_ANCHOR_CHARS = 200;
/**
 * Per-page caps on the extracted collections. Crawler-trap and mega-menu
 * pages can carry thousands of links/images per page, and crawled pages sit
 * in memory in 25-page persist batches — uncapped collections were part of
 * the audit engine's exceededMemory profile. Counts derived from these
 * arrays saturate at the cap on such pathological pages.
 */
const MAX_EXTRACTED_LINKS = 1_000;
const MAX_EXTRACTED_IMAGES = 1_000;
// Skip oversized blocks whole: truncation would manufacture JSON errors.
const MAX_JSON_LD_BLOCK_CHARS = 100_000;
const MAX_JSON_LD_BLOCKS = 20;
const MAX_JSON_LD_NODES = 200;
const jsonLdObjectSchema = z.record(z.string(), z.unknown());
// Google's required properties per rich-result type. Article and Organization
// have only recommended properties, so they are not checked.
const REQUIRED_JSON_LD_PROPERTIES: Record<string, readonly string[]> = {
  LocalBusiness: ["name", "address"],
  Product: ["name"],
  FAQPage: ["mainEntity"],
  BreadcrumbList: ["itemListElement"],
  Event: ["name", "startDate", "location"],
  Recipe: ["name", "image"],
};

function hasJsonLdValue(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return true;
}

// Common app entry points. Scripts alone also appear on ordinary HTML pages.
const APP_ROOT_IDS = new Set(["root", "app", "__next", "__nuxt"]);
/**
 * Elements a browser renders on their own line. Minified HTML has no
 * whitespace between them, so `<p>one</p><p>two</p>` would read as "onetwo"
 * without a separator at their boundaries.
 */
const BLOCK_TAGS = new Set([
  "address",
  "article",
  "aside",
  "blockquote",
  "br",
  "caption",
  "dd",
  "details",
  "dialog",
  "div",
  "dl",
  "dt",
  "fieldset",
  "figcaption",
  "figure",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "header",
  "hgroup",
  "hr",
  "li",
  "main",
  "nav",
  "ol",
  "p",
  "pre",
  "section",
  "summary",
  "table",
  "td",
  "th",
  "tr",
  "ul",
]);
// Chinese, Japanese, Thai, Lao, Khmer and Burmese are written without spaces
// between words, so a whitespace split counts a whole paragraph as one word
// and flags those pages as thin content. Tokens in those scripts get their
// words from Intl.Segmenter; every other token still counts as one word, so
// counts for space-separated languages are unchanged.
const SPACELESS_SCRIPT =
  /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]/u;
const wordSegmenter = new Intl.Segmenter(undefined, { granularity: "word" });

function countWords(text: string): number {
  let count = 0;
  for (const token of text.split(/\s+/)) {
    if (!token) continue;
    if (!SPACELESS_SCRIPT.test(token)) {
      count += 1;
      continue;
    }
    for (const segment of wordSegmenter.segment(token)) {
      if (segment.isWordLike) count += 1;
    }
  }
  return count;
}

interface OpenAnchor {
  href: string;
  rel: string;
  text: string[];
}

/**
 * Analyze an HTML string and extract all SEO-relevant data.
 */
export function analyzeHtml(
  html: string,
  pageUrl: string,
  statusCode: number,
  responseTimeMs: number,
  redirectUrl: string | null = null,
): PageAnalysis {
  let title: string | null = null;
  let titleDepth = 0;
  let titleDone = false;
  // parse5 (the old DOM path) treats <noscript> content as raw text when
  // scripting is enabled; skip element extraction inside it to match.
  let noscriptDepth = 0;
  let metaDescription: string | null = null;
  let canonical: string | null = null;
  let robotsMeta: string | null = null;
  let ogTitle: string | null = null;
  let ogDescription: string | null = null;
  let ogImage: string | null = null;
  const structuredDataIssues: PageAnalysis["structuredDataIssues"] = [];
  let jsonLdText: string | null = null;
  let jsonLdBlocks = 0;
  let jsonLdNodes = 0;
  let hasAppRoot = false;
  let hasExecutableScript = false;
  const hreflangTags: string[] = [];

  const h1s: string[] = [];
  const headingOrder: number[] = [];
  let openH1: string[] | null = null;

  const images: Array<{ src: string | null; alt: string | null }> = [];
  const linksByTarget = new Map<string, PageLink>();
  let openAnchor: OpenAnchor | null = null;

  // Visible text: prefer text inside an explicit <body>; when the document
  // never opens one (fragments), fall back to all non-head text. Both
  // exclude NON_CONTENT_TAGS subtrees.
  let suppressDepth = 0;
  let bodyDepth = 0;
  let headDepth = 0;
  let sawBody = false;
  const bodyParts: string[] = [];
  const fallbackParts: string[] = [];

  const validateJsonLd = (text: string, block: number) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      structuredDataIssues.push({
        issueType: "invalid-json-ld",
        details: { block },
      });
      return;
    }

    // Only document and @graph nodes need a type. Property values can be
    // untyped value objects or @id references, so don't walk arbitrary keys.
    const pending: unknown[] = [parsed];
    let object = 0;
    while (pending.length > 0 && jsonLdNodes < MAX_JSON_LD_NODES) {
      const value = pending.pop();
      jsonLdNodes += 1;
      if (Array.isArray(value)) {
        const count = Math.min(
          value.length,
          MAX_JSON_LD_NODES - jsonLdNodes - pending.length,
        );
        for (let i = count - 1; i >= 0; i--) pending.push(value[i]);
        continue;
      }
      const result = jsonLdObjectSchema.safeParse(value);
      if (!result.success) continue;
      const node = result.data;
      object += 1;
      const rawTypes = node["@type"];
      const types = (Array.isArray(rawTypes) ? rawTypes : [rawTypes]).filter(
        (type): type is string => typeof type === "string" && !!type.trim(),
      );
      if (types.length === 0 && !("@graph" in node)) {
        structuredDataIssues.push({
          issueType: "json-ld-missing-type",
          details: { block, object },
        });
      }
      for (const type of new Set(types)) {
        if (!Object.hasOwn(REQUIRED_JSON_LD_PROPERTIES, type)) continue;
        const required = REQUIRED_JSON_LD_PROPERTIES[type];
        const missingProperties = required.filter(
          (property) => !hasJsonLdValue(node[property]),
        );
        if (
          type === "Product" &&
          !["offers", "review", "aggregateRating"].some((property) =>
            hasJsonLdValue(node[property]),
          )
        ) {
          missingProperties.push("offers/review/aggregateRating");
        }
        if (missingProperties.length > 0) {
          structuredDataIssues.push({
            issueType: "json-ld-missing-required-properties",
            details: { block, object, type, missingProperties },
          });
        }
      }
      if ("@graph" in node) pending.push(node["@graph"]);
    }
  };

  const addVisibleText = (text: string) => {
    if (suppressDepth > 0) return;
    if (bodyDepth > 0) {
      bodyParts.push(text);
    } else if (headDepth === 0) {
      fallbackParts.push(text);
    }
  };

  const handleMetaTag = (attribs: Record<string, string>) => {
    const content = attribs["content"];
    // HTML meta names are ASCII case-insensitive: "Description" is valid.
    const metaName = attribs["name"]?.toLowerCase();
    if (metaName === "description") {
      metaDescription ??= content?.trim() ?? "";
    } else if (metaName === "robots") {
      robotsMeta ??= content ?? null;
    } else if (attribs["property"] === "og:title") {
      ogTitle ??= content ?? null;
    } else if (attribs["property"] === "og:description") {
      ogDescription ??= content ?? null;
    } else if (attribs["property"] === "og:image") {
      ogImage ??= content ?? null;
    }
  };

  const handleLinkTag = (attribs: Record<string, string>) => {
    if (attribs["rel"] === "canonical") {
      canonical ??= attribs["href"] ?? null;
    } else if (attribs["rel"] === "alternate" && attribs["hreflang"]) {
      hreflangTags.push(attribs["hreflang"]);
    }
  };

  const closeAnchor = () => {
    if (!openAnchor) return;
    const { href, rel, text } = openAnchor;
    openAnchor = null;
    if (linksByTarget.size >= MAX_EXTRACTED_LINKS) return;
    const resolved = normalizeUrl(href, pageUrl);
    if (!resolved || linksByTarget.has(resolved)) return;
    const anchor = text
      .join("")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, MAX_ANCHOR_CHARS);
    linksByTarget.set(resolved, {
      targetUrl: resolved,
      anchor: anchor || null,
      isInternal: isSameOrigin(resolved, pageUrl),
      isNofollow: rel.split(/\s+/).includes("nofollow"),
    });
  };

  const parser = new Parser(
    {
      onopentag(name, attribs) {
        if (NON_CONTENT_TAGS.has(name)) {
          suppressDepth += 1;
        }
        if (name === "noscript") noscriptDepth += 1;
        if (noscriptDepth > 0) return;
        if (BLOCK_TAGS.has(name)) addVisibleText(" ");
        if (headDepth === 0 && suppressDepth === 0) {
          hasAppRoot ||= APP_ROOT_IDS.has(attribs["id"]) || name === "app-root";
        }
        switch (name) {
          case "title":
            // Ignore <title> inside <svg> — only the document title counts.
            if (!titleDone && suppressDepth === 0) {
              titleDepth += 1;
              if (title === null) title = "";
            }
            break;
          case "head":
            headDepth += 1;
            break;
          case "body":
            bodyDepth += 1;
            sawBody = true;
            break;
          case "meta":
            handleMetaTag(attribs);
            break;
          case "link":
            handleLinkTag(attribs);
            break;
          case "img":
            if (images.length < MAX_EXTRACTED_IMAGES) {
              images.push({
                src: attribs["src"] ?? null,
                alt: "alt" in attribs ? attribs["alt"] : null,
              });
            }
            break;
          case "script":
            hasExecutableScript ||= [
              "",
              "module",
              "text/javascript",
              "application/javascript",
            ].includes(attribs["type"]?.toLowerCase() ?? "");
            if (attribs["type"] === "application/ld+json") {
              jsonLdBlocks += 1;
              if (jsonLdBlocks <= MAX_JSON_LD_BLOCKS) jsonLdText = "";
            }
            break;
          case "a": {
            // HTML forbids nested <a>; browsers implicitly close the open
            // one, and the tokenizer has no tree correction, so mirror that.
            closeAnchor();
            const href = attribs["href"];
            if (href && !SKIPPED_LINK_PROTOCOLS.test(href)) {
              openAnchor = {
                href,
                rel: attribs["rel"]?.toLowerCase() ?? "",
                text: [],
              };
            }
            break;
          }
        }
        const headingLevel = HEADING_LEVELS[name];
        if (headingLevel !== undefined) {
          headingOrder.push(headingLevel);
          if (headingLevel === 1 && openH1 === null) openH1 = [];
        }
      },
      ontext(text) {
        if (jsonLdText !== null) {
          jsonLdText =
            jsonLdText.length + text.length <= MAX_JSON_LD_BLOCK_CHARS
              ? jsonLdText + text
              : null;
        }
        if (suppressDepth > 0) return;
        if (titleDepth > 0) {
          if (title !== null) title += text;
          return;
        }
        if (openH1) openH1.push(text);
        if (openAnchor) openAnchor.text.push(text);
        addVisibleText(text);
      },
      onclosetag(name) {
        if (name === "script" && jsonLdText !== null) {
          validateJsonLd(jsonLdText, jsonLdBlocks);
          jsonLdText = null;
        }
        if (NON_CONTENT_TAGS.has(name) && suppressDepth > 0) {
          suppressDepth -= 1;
        }
        if (name === "noscript" && noscriptDepth > 0) {
          noscriptDepth -= 1;
          return;
        }
        if (noscriptDepth > 0) return;
        if (BLOCK_TAGS.has(name)) addVisibleText(" ");
        if (name === "title" && titleDepth > 0) {
          titleDepth -= 1;
          if (titleDepth === 0) titleDone = true;
        }
        if (name === "head" && headDepth > 0) headDepth -= 1;
        if (name === "body" && bodyDepth > 0) bodyDepth -= 1;
        if (name === "a") closeAnchor();
        if (name === "h1" && openH1) {
          h1s.push(openH1.join("").trim());
          openH1 = null;
        }
      },
    },
    // Defaults (non-XML mode): lowercased tag/attribute names, decoded
    // entities — matching what the DOM-based implementation saw.
  );
  parser.write(html);
  parser.end();

  const rawText = (sawBody ? bodyParts : fallbackParts).join("");
  const bodyText = rawText.replace(/\s+/g, " ").trim();
  const wordCount = countWords(bodyText);

  return {
    url: pageUrl,
    statusCode,
    redirectUrl,
    responseTimeMs,
    title: (title ?? "").trim(),
    metaDescription: metaDescription ?? "",
    canonical,
    robotsMeta,
    ogTitle,
    ogDescription,
    ogImage,
    h1s,
    headingOrder,
    wordCount,
    bodyText,
    javascriptShell:
      hasAppRoot &&
      hasExecutableScript &&
      wordCount < 20 &&
      headingOrder.length === 0 &&
      linksByTarget.size === 0 &&
      images.length === 0,
    images,
    links: Array.from(linksByTarget.values()),
    hasStructuredData: jsonLdBlocks > 0,
    structuredDataIssues,
    hreflangTags,
  };
}
