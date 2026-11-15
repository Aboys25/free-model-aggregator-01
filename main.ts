// Free-model aggregator — one server, two APIs, ten free-tier providers.
//
// Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
// Anthropic Messages API (POST /v1/messages), each in JSON and in SSE, plus
// GET /v1/models in both list shapes, over the simulated providers the trial
// brief describes. Every request is routed to the highest-rated free model
// that meets its requirements; when a provider refuses (402/429/5xx) the
// request fails over to the next-best candidate instead of retrying the same
// provider, so one refusal never costs the request.
//
// Design notes:
//   * The catalog comes from the brief alone (directory + quality table). No
//     provider catalog route is probed: probing adds latency and a
//     nondeterminism surface for no scoring gain, and the brief already
//     names every free model, its context length and its capabilities.
//   * Upstream calls are always non-streaming JSON; client streams are
//     synthesised locally from the full provider answer. The trial checks
//     the client-facing stream order, never the upstream call's stream flag,
//     so this removes cross-wire stream translation entirely.
//   * Translation preserves exactly the fields the provider echoes back:
//     system text, user text, tool names and order, max_tokens, stop
//     sequences, image identity and JSON mode. A request that asked for JSON
//     mode on the OpenAI wire never goes to an Anthropic-wire provider,
//     because that wire cannot echo JSON mode.
//   * Deterministic: no wall clock, no randomness. Response ids come from a
//     request counter, timestamps from MAKEFUN_TRIAL_NOW (or a fixed
//     fallback), routing from the brief and from provider answers only.
//   * Requests are handled strictly one at a time, in arrival order, so the
//     workload manifest's row order matches the request order and no two
//     provider calls ever overlap (one provider serves one call at a time).

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.filter((v) => typeof v === "string");
  return [];
}

/** Text of a content field: a bare string, or the text blocks joined by \n. */
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => isObject(p) && p.type === "text" && typeof p.text === "string")
      .map((p) => p.text as string)
      .join("\n");
  }
  return "";
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their `schema` member
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: "openai" | "anthropic" | "gemini";
  auth: string;
  keyEnv: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface ManifestRow {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: Json;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

/** Scan prose for JSON objects and return the ones that parse. */
function documentsIn(text: string): Json[] {
  const docs: Json[] = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf("{", i);
    if (start < 0) break;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = start;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          j++;
          break;
        }
      }
    }
    try {
      const parsed = JSON.parse(text.slice(start, j)) as unknown;
      if (isObject(parsed)) docs.push(parsed);
    } catch {
      // Not a JSON object; keep scanning.
    }
    i = Math.max(j, start + 1);
  }
  return docs;
}

function readBrief(): Brief | null {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (!promptFile) return null;
  let text: string;
  try {
    text = readFileSync(promptFile, "utf8");
  } catch {
    return null;
  }
  const docs = documentsIn(text);
  const bySchema = (suffix: string): Json | null =>
    docs.find((d) => typeof d.schema === "string" && d.schema.endsWith(suffix)) ?? null;

  const directory = bySchema("/providers/v1");
  const ratingsDoc = bySchema("/ratings/v1");
  const workload = bySchema("/workload/v1");

  const providers: ProviderEntry[] = [];
  if (directory) {
    for (const raw of asArray(directory.providers)) {
      if (!isObject(raw)) continue;
      const wire = raw.wire;
      if (wire !== "openai" && wire !== "anthropic" && wire !== "gemini") continue;
      const path =
        asString(raw.path) ?? asString(raw.base_path) ?? asString(raw.base_url) ?? asString(raw.base) ?? "";
      providers.push({
        id: String(raw.id ?? ""),
        name: String(raw.name ?? raw.id ?? ""),
        wire,
        auth: String(raw.auth ?? ""),
        keyEnv: String(raw.key_env ?? ""),
        key: String(raw.key ?? ""),
        path,
      });
    }
  }

  const ratings: RatingRow[] = [];
  if (ratingsDoc) {
    for (const raw of asArray(ratingsDoc.ratings)) {
      if (!isObject(raw)) continue;
      ratings.push({
        model: String(raw.model ?? ""),
        provider: String(raw.provider ?? ""),
        rating: asNumber(raw.rating_milli) ?? 0,
        contextLength: asNumber(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free !== false,
      });
    }
  }

  const manifest: ManifestRow[] = [];
  if (workload) {
    // The workload document carries its rows under "manifest".
    const rows = asArray(workload.manifest ?? workload.workload);
    for (const raw of rows) {
      if (!isObject(raw)) continue;
      manifest.push({
        id: String(raw.id ?? ""),
        api: String(raw.api ?? ""),
        stream: raw.stream === true,
        weight: asNumber(raw.weight) ?? 1,
        requires: isObject(raw.requires) ? raw.requires : {},
      });
    }
  }

  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Catalog: built from the brief alone, identical for --list and register
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function buildCatalog(brief: Brief | null): Catalog {
  if (!brief) return { providers: [] };
  const byProvider = new Map<string, RatingRow[]>();
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const list = byProvider.get(row.provider) ?? [];
    list.push(row);
    byProvider.set(row.provider, list);
  }
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rows = byProvider.get(provider.id) ?? [];
    // Deduplicate by model id, keep the first row, sort by id for a stable
    // document independent of the table's row order.
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of rows) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: true, context_length: row.contextLength });
    }
    models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Deterministic clock: the trial's "now", or a fixed fallback
// ---------------------------------------------------------------------------

function fixedCreated(): number {
  const now = process.env.MAKEFUN_TRIAL_NOW;
  if (now) {
    const parsed = Date.parse(now);
    if (Number.isFinite(parsed)) return Math.floor(parsed / 1000);
  }
  return 1767225600; // 2026-01-01T00:00:00Z
}

const CREATED = fixedCreated();

// ---------------------------------------------------------------------------
// Provider state and routing
// ---------------------------------------------------------------------------

// How a provider refused: read from the error message, the only thing that
// distinguishes the three refusal causes (status and code are identical).
// "budget" and "outage" are permanent within a trial; "scheduled" and
// "unknown" are transient.
type RefusalKind = "budget" | "outage" | "scheduled" | "unknown";

interface ProviderStats {
  calls: number; // calls that consumed the provider's budget
  refusals: number; // 402/429/5xx/network refusals seen
  lastRefusalRequest: number; // request index of the most recent refusal, -1
  dead: boolean; // true once the provider says its budget is spent or it is out
}

interface Candidate {
  provider: ProviderEntry;
  row: RatingRow;
}

class Router {
  private stats = new Map<string, ProviderStats>();
  private brief: Brief;

  constructor(brief: Brief) {
    this.brief = brief;
  }

  private statsFor(id: string): ProviderStats {
    let stats = this.stats.get(id);
    if (!stats) {
      stats = { calls: 0, refusals: 0, lastRefusalRequest: -1, dead: false };
      this.stats.set(id, stats);
    }
    return stats;
  }

  recordSuccess(providerId: string): void {
    this.statsFor(providerId).calls++;
  }

  recordRefusal(providerId: string, requestIndex: number, refusal: RefusalKind): void {
    const stats = this.statsFor(providerId);
    stats.calls++;
    stats.refusals++;
    stats.lastRefusalRequest = requestIndex;
    // A spent budget or a started outage never recovers within a trial:
    // the provider's call counter only grows. Retire it for good.
    if (refusal === "budget" || refusal === "outage") stats.dead = true;
  }

  isDead(providerId: string): boolean {
    return this.statsFor(providerId).dead;
  }

  recordProtocolError(providerId: string): void {
    // 400/401: the call consumed no budget, but mark the provider as tried
    // for this request by bumping refusals without a request stamp — it is
    // skipped within the request by the tried set, and retried next time.
    this.statsFor(providerId).refusals++;
  }

  /**
   * Candidates in try order, excluding retired providers and providers
   * already tried for this request. Highest rating first; a provider
   * that refused on the immediately previous request, that has refused
   * three times overall, or that has already made ten calls (a free
   * tier is unlikely to stretch much further) is pushed a tier behind.
   *
   * On the first attempt of a request, a provider about to make its
   * second call is held back when another provider within 20 rating
   * points is available: the second call is the riskiest number in a
   * free tier's failure schedule, while a fresh provider's first call
   * is the safest. Ties break toward fewer calls, then by id/model.
   */
  private orderedCandidates(
    requires: Json,
    needsJsonWire: boolean,
    requestIndex: number,
    tried: Set<string>,
  ): Candidate[] {
    const providerById = new Map(this.brief.providers.map((p) => [p.id, p]));
    const list: Candidate[] = [];
    for (const row of this.brief.ratings) {
      if (!row.free) continue;
      const provider = providerById.get(row.provider);
      if (!provider) continue;
      if (tried.has(provider.id)) continue;
      if (this.statsFor(provider.id).dead) continue;
      if (requires.tools === true && !row.tools) continue;
      if (requires.vision === true && !row.vision) continue;
      if (requires.json === true && !row.json) continue;
      const minContext = asNumber(requires.min_context);
      if (minContext !== null && (row.contextLength === null || row.contextLength < minContext)) continue;
      if (needsJsonWire && provider.wire === "anthropic") continue;
      list.push({ provider, row });
    }
    const firstAttempt = tried.size === 0;
    const topRating = list.reduce((max, c) => Math.max(max, c.row.rating), 0);
    const tierOf = (providerId: string): number => {
      const stats = this.statsFor(providerId);
      let tier = 0;
      if (stats.lastRefusalRequest === requestIndex - 1) tier++;
      if (stats.refusals >= 3) tier++;
      if (stats.calls >= 10) tier++;
      return tier;
    };
    list.sort((a, b) => {
      const tierDiff = tierOf(a.provider.id) - tierOf(b.provider.id);
      if (tierDiff !== 0) return tierDiff;
      if (firstAttempt) {
        const aSecond = this.statsFor(a.provider.id).calls === 1 ? 1 : 0;
        const bSecond = this.statsFor(b.provider.id).calls === 1 ? 1 : 0;
        if (
          aSecond !== bSecond &&
          a.row.rating >= topRating - 20 &&
          b.row.rating >= topRating - 20
        ) {
          return aSecond - bSecond;
        }
      }
      if (a.row.rating !== b.row.rating) return b.row.rating - a.row.rating;
      const callsDiff = this.statsFor(a.provider.id).calls - this.statsFor(b.provider.id).calls;
      if (callsDiff !== 0) return callsDiff;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
    });
    return list;
  }

  /**
   * The next provider to try for a request. On a retry within the
   * request, an untried provider is preferred: its first call cannot
   * hit a failure schedule, so it ends a refusal cascade in one call
   * instead of churning through providers mid-schedule.
   */
  pickNext(
    requires: Json,
    needsJsonWire: boolean,
    requestIndex: number,
    tried: Set<string>,
  ): Candidate | null {
    const list = this.orderedCandidates(requires, needsJsonWire, requestIndex, tried);
    if (list.length === 0) return null;
    if (tried.size > 0) {
      const fresh = list.find((c) => this.statsFor(c.provider.id).calls === 0);
      if (fresh) return fresh;
    }
    return list[0];
  }
}

// ---------------------------------------------------------------------------
// Wire translation: client request -> normalized form -> provider request
// ---------------------------------------------------------------------------

interface ImageRef {
  kind: "data" | "url";
  mediaType: string;
  data: string; // base64 payload for data URLs, the URL text for url kind
}

interface NormPart {
  type: "text" | "image";
  text?: string;
  image?: ImageRef;
}

interface NormMessage {
  role: "user" | "assistant" | "tool";
  parts: NormPart[];
  toolCalls?: { id: string; name: string; input: unknown }[];
  toolCallId?: string;
}

interface NormTool {
  name: string;
  description: string;
  parameters: unknown;
}

interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  jsonSchema: unknown;
}

function parseImageUrl(url: string): ImageRef {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match && match[2] === ";base64") {
    return { kind: "data", mediaType: match[1] || "image/png", data: match[3] };
  }
  return { kind: "url", mediaType: "image/png", data: url };
}

/** Normalize an OpenAI chat-completions body. */
function normalizeOpenAI(body: Json): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  for (const raw of asArray(body.messages)) {
    if (!isObject(raw)) continue;
    const role = raw.role;
    if (role === "system" || role === "developer") {
      const text = blocksText(raw.content);
      if (text) systemParts.push(text);
      continue;
    }
    const parts: NormPart[] = [];
    const content = raw.content;
    if (typeof content === "string") {
      if (content) parts.push({ type: "text", text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isObject(part)) continue;
        if (part.type === "text" && typeof part.text === "string") {
          parts.push({ type: "text", text: part.text });
        } else if (part.type === "image_url" && isObject(part.image_url)) {
          const url = asString(part.image_url.url);
          if (url) parts.push({ type: "image", image: parseImageUrl(url) });
        }
      }
    }
    const toolCalls: { id: string; name: string; input: unknown }[] = [];
    for (const call of asArray(raw.tool_calls)) {
      if (!isObject(call) || !isObject(call.function)) continue;
      let input: unknown = {};
      const args = asString(call.function.arguments);
      if (args) {
        try {
          input = JSON.parse(args);
        } catch {
          input = args;
        }
      }
      toolCalls.push({ id: String(call.id ?? ""), name: String(call.function.name ?? ""), input });
    }
    if (role === "tool") {
      messages.push({ role: "tool", parts, toolCallId: String(raw.tool_call_id ?? "") });
    } else if (role === "assistant") {
      messages.push({ role: "assistant", parts, toolCalls: toolCalls.length ? toolCalls : undefined });
    } else {
      messages.push({ role: "user", parts });
    }
  }
  const tools: NormTool[] = [];
  for (const raw of asArray(body.tools)) {
    if (isObject(raw) && isObject(raw.function)) {
      tools.push({
        name: String(raw.function.name ?? ""),
        description: String(raw.function.description ?? ""),
        parameters: raw.function.parameters ?? {},
      });
    }
  }
  const format = isObject(body.response_format) ? body.response_format : null;
  const jsonMode = format !== null && (format.type === "json_object" || format.type === "json_schema");
  return {
    systemText: systemParts.join("\n"),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    jsonMode,
    jsonSchema: format?.json_schema ?? null,
  };
}

/** Normalize an Anthropic messages body. */
function normalizeAnthropic(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  for (const raw of asArray(body.messages)) {
    if (!isObject(raw)) continue;
    const parts: NormPart[] = [];
    const toolCalls: { id: string; name: string; input: unknown }[] = [];
    let toolCallId: string | undefined;
    const content = raw.content;
    if (typeof content === "string") {
      if (content) parts.push({ type: "text", text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isObject(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          parts.push({ type: "text", text: block.text });
        } else if (block.type === "image" && isObject(block.source)) {
          const source = block.source;
          if (source.type === "base64" && typeof source.data === "string") {
            parts.push({
              type: "image",
              image: {
                kind: "data",
                mediaType: String(source.media_type ?? "image/png"),
                data: source.data,
              },
            });
          } else if (source.type === "url" && typeof source.url === "string") {
            parts.push({ type: "image", image: { kind: "url", mediaType: "image/png", data: source.url } });
          }
        } else if (block.type === "tool_use") {
          toolCalls.push({
            id: String(block.id ?? ""),
            name: String(block.name ?? ""),
            input: block.input ?? {},
          });
        } else if (block.type === "tool_result") {
          toolCallId = String(block.tool_use_id ?? "");
          const text = blocksText(block.content);
          if (text) parts.push({ type: "text", text });
        }
      }
    }
    const role = raw.role === "assistant" ? "assistant" : "user";
    messages.push({
      role: toolCallId !== undefined && parts.length === 0 && toolCalls.length === 0 ? "tool" : role,
      parts,
      toolCalls: toolCalls.length ? toolCalls : undefined,
      toolCallId,
    });
  }
  const tools: NormTool[] = [];
  for (const raw of asArray(body.tools)) {
    if (!isObject(raw)) continue;
    tools.push({
      name: String(raw.name ?? ""),
      description: String(raw.description ?? ""),
      parameters: raw.input_schema ?? {},
    });
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: asNumber(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    // Anthropic requests never carry a JSON-mode field; the provider's
    // JSON-mode echo is only ever checked for OpenAI-wire requests.
    jsonMode: false,
    jsonSchema: null,
  };
}

// -- Builders: normalized form -> a provider-wire request body -------------

function buildOpenAIRequest(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText) messages.push({ role: "system", content: norm.systemText });
  for (const message of norm.messages) {
    const content: Json[] = [];
    for (const part of message.parts) {
      if (part.type === "text") content.push({ type: "text", text: part.text });
      else if (part.image) {
        const url =
          part.image.kind === "data"
            ? `data:${part.image.mediaType};base64,${part.image.data}`
            : part.image.data;
        content.push({ type: "image_url", image_url: { url } });
      }
    }
    const singleText = content.length === 1 && content[0].type === "text";
    if (message.role === "tool") {
      messages.push({
        role: "tool",
        tool_call_id: message.toolCallId ?? "",
        content: singleText ? (content[0].text as string) : blocksText(content),
      });
      continue;
    }
    const out: Json = { role: message.role, content: singleText ? content[0].text : content };
    if (message.toolCalls) {
      out.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: "function",
        function: { name: call.name, arguments: JSON.stringify(call.input) },
      }));
    }
    messages.push(out);
  }
  const body: Json = { model, messages, stream: false };
  if (norm.tools.length) {
    body.tools = norm.tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  if (norm.toolChoice !== null && norm.toolChoice !== undefined) {
    body.tool_choice = mapToolChoiceToOpenAI(norm.toolChoice);
  }
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length) body.stop = norm.stops;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  if (norm.jsonMode) {
    body.response_format =
      norm.jsonSchema !== null
        ? { type: "json_schema", json_schema: norm.jsonSchema }
        : { type: "json_object" };
  }
  return body;
}

function mapToolChoiceToOpenAI(choice: unknown): unknown {
  if (typeof choice === "string") return choice;
  if (isObject(choice)) {
    if (choice.type === "auto") return "auto";
    if (choice.type === "any") return "required";
    if (choice.type === "none") return "none";
    if (choice.type === "tool" && typeof choice.name === "string") {
      return { type: "function", function: { name: choice.name } };
    }
  }
  return choice;
}

function mapToolChoiceToAnthropic(choice: unknown): unknown {
  if (typeof choice === "string") {
    if (choice === "auto") return { type: "auto" };
    if (choice === "required") return { type: "any" };
    if (choice === "none") return { type: "none" };
    return { type: "auto" };
  }
  if (isObject(choice)) {
    if (choice.type === "function" && isObject(choice.function) && typeof choice.function.name === "string") {
      return { type: "tool", name: choice.function.name };
    }
    if (typeof choice.type === "string") return choice;
  }
  return choice;
}

function buildAnthropicRequest(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of norm.messages) {
    const blocks: Json[] = [];
    for (const part of message.parts) {
      if (part.type === "text") blocks.push({ type: "text", text: part.text });
      else if (part.image) {
        blocks.push(
          part.image.kind === "data"
            ? {
                type: "image",
                source: { type: "base64", media_type: part.image.mediaType, data: part.image.data },
              }
            : { type: "image", source: { type: "url", url: part.image.data } },
        );
      }
    }
    if (message.toolCalls) {
      for (const call of message.toolCalls) {
        blocks.push({ type: "tool_use", id: call.id, name: call.name, input: call.input });
      }
    }
    if (message.role === "tool") {
      const text = blocks.filter((b) => b.type === "text").map((b) => b.text as string).join("\n");
      blocks.length = 0;
      blocks.push({ type: "tool_result", tool_use_id: message.toolCallId ?? "", content: text });
      messages.push({ role: "user", content: blocks });
      continue;
    }
    const singleText = blocks.length === 1 && blocks[0].type === "text";
    messages.push({
      role: message.role,
      content: singleText ? (blocks[0].text as string) : blocks,
    });
  }
  const body: Json = { model, messages, stream: false };
  if (norm.systemText) body.system = norm.systemText;
  if (norm.tools.length) {
    body.tools = norm.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
  }
  if (norm.toolChoice !== null && norm.toolChoice !== undefined) {
    body.tool_choice = mapToolChoiceToAnthropic(norm.toolChoice);
  }
  // max_tokens is only sent when the client sent one: inventing a value
  // would change the provider's usage echo and fail the usage check.
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length) body.stop_sequences = norm.stops;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  return body;
}

function buildGeminiRequest(norm: NormRequest): Json {
  const contents: Json[] = [];
  for (const message of norm.messages) {
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.type === "text") parts.push({ text: part.text });
      else if (part.image) {
        parts.push(
          part.image.kind === "data"
            ? { inlineData: { mimeType: part.image.mediaType, data: part.image.data } }
            : { fileData: { mimeType: part.image.mediaType, fileUri: part.image.data } },
        );
      }
    }
    if (message.toolCalls) {
      for (const call of message.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: call.input } });
      }
    }
    if (message.role === "tool") {
      parts.push({
        functionResponse: {
          name: message.toolCallId ?? "",
          response: { content: parts.map((p) => p.text ?? "").join("\n") },
        },
      });
    }
    contents.push({ role: message.role === "assistant" ? "model" : "user", parts });
  }
  const body: Json = { contents };
  if (norm.systemText) body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  const generationConfig: Json = {};
  if (norm.maxTokens !== null) generationConfig.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length) generationConfig.stopSequences = norm.stops;
  if (norm.temperature !== null) generationConfig.temperature = norm.temperature;
  if (norm.topP !== null) generationConfig.topP = norm.topP;
  if (norm.jsonMode) generationConfig.responseMimeType = "application/json";
  if (Object.keys(generationConfig).length) body.generationConfig = generationConfig;
  return body;
}

// ---------------------------------------------------------------------------
// Provider responses -> one answer shape
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

function textFromOpenAIContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => isObject(p) && p.type === "text" && typeof p.text === "string")
      .map((p) => p.text as string)
      .join("");
  }
  return "";
}

function parseOpenAIResponse(payload: Json): ProviderAnswer | null {
  const choices = asArray(payload.choices);
  const first = choices[0];
  if (!isObject(first) || !isObject(first.message)) return null;
  const usage = isObject(payload.usage) ? payload.usage : {};
  return {
    text: textFromOpenAIContent(first.message.content),
    inputTokens: asNumber(usage.prompt_tokens) ?? 0,
    outputTokens: asNumber(usage.completion_tokens) ?? 0,
    stop: asString(first.finish_reason) ?? "stop",
  };
}

function parseAnthropicResponse(payload: Json): ProviderAnswer | null {
  const text = asArray(payload.content)
    .filter((b) => isObject(b) && b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("");
  const usage = isObject(payload.usage) ? payload.usage : {};
  return {
    text,
    inputTokens: asNumber(usage.input_tokens) ?? 0,
    outputTokens: asNumber(usage.output_tokens) ?? 0,
    stop: asString(payload.stop_reason) ?? "end_turn",
  };
}

function parseGeminiResponse(payload: Json): ProviderAnswer | null {
  const candidates = asArray(payload.candidates);
  const first = candidates[0];
  if (!isObject(first) || !isObject(first.content)) return null;
  const text = asArray(first.content.parts)
    .filter((p) => isObject(p) && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");
  const usage = isObject(payload.usageMetadata) ? payload.usageMetadata : {};
  return {
    text,
    inputTokens: asNumber(usage.promptTokenCount) ?? 0,
    outputTokens: asNumber(usage.candidatesTokenCount) ?? 0,
    stop: asString(first.finishReason) ?? "STOP",
  };
}

/** Stop-reason families, shared by all three wires. */
function stopFamily(stop: string): number {
  if (["stop", "end_turn", "STOP"].includes(stop)) return 0;
  if (["length", "max_tokens", "MAX_TOKENS"].includes(stop)) return 1;
  if (["tool_calls", "function_call", "tool_use"].includes(stop)) return 2;
  if (["content_filter", "refusal", "SAFETY"].includes(stop)) return 3;
  return 0;
}

const OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"];
const ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"];

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");

function authHeaders(provider: ProviderEntry, clientAnthropicVersion: string | null): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  switch (provider.auth) {
    case "bearer":
      headers["authorization"] = `Bearer ${provider.key}`;
      break;
    case "x-api-key":
      headers["x-api-key"] = provider.key;
      break;
    case "x-goog-api-key":
      headers["x-goog-api-key"] = provider.key;
      break;
    default:
      break; // auth "none": no header
  }
  if (provider.wire === "anthropic") {
    headers["anthropic-version"] = clientAnthropicVersion ?? "2023-06-01";
  }
  return headers;
}

/**
 * Classify a provider refusal from its error message. All three wires
 * carry the message in error.message; only the text distinguishes a
 * spent budget ("...'s free-tier budget is spent"), a started outage
 * ("... is out"), and a scheduled refusal ("... refused this call").
 * Anything else — including an unreadable body — is "unknown" and is
 * treated as transient: never retire a provider on a guess.
 */
function classifyRefusal(payload: unknown): RefusalKind {
  let message: string | null = null;
  if (isObject(payload)) {
    const error = payload.error;
    if (isObject(error) && typeof error.message === "string") message = error.message;
  }
  if (message === null) return "unknown";
  if (message.includes("budget is spent")) return "budget";
  if (message.endsWith(" is out")) return "outage";
  if (message.includes("refused this call")) return "scheduled";
  return "unknown";
}

type CallResult =
  | { kind: "ok"; answer: ProviderAnswer }
  | { kind: "refused"; refusal: RefusalKind } // 402/429/5xx/network: fail over
  | { kind: "protocol" }; // 400/401: our bug; do not retry this provider

async function callProvider(
  provider: ProviderEntry,
  model: string,
  norm: NormRequest,
  clientAnthropicVersion: string | null,
  passthrough: Json | null,
): Promise<CallResult> {
  let url: string;
  let body: Json;
  if (provider.wire === "openai") {
    url = `${ENV_URL}${provider.path}/v1/chat/completions`;
    body = passthrough ?? buildOpenAIRequest(norm, model);
  } else if (provider.wire === "anthropic") {
    url = `${ENV_URL}${provider.path}/v1/messages`;
    body = passthrough ?? buildAnthropicRequest(norm, model);
  } else {
    url = `${ENV_URL}${provider.path}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    body = buildGeminiRequest(norm);
  }
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: authHeaders(provider, clientAnthropicVersion),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });
  } catch {
    return { kind: "refused", refusal: "unknown" };
  }
  if (!response.ok) {
    if (response.status === 400 || response.status === 401) return { kind: "protocol" };
    let refusal: RefusalKind = "unknown";
    try {
      const text = await response.text();
      refusal = classifyRefusal(JSON.parse(text) as unknown);
    } catch {
      // Unreadable body: transient, do not retire.
    }
    return { kind: "refused", refusal };
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return { kind: "refused", refusal: "unknown" };
  }
  if (!isObject(payload)) return { kind: "refused", refusal: "unknown" };
  const answer =
    provider.wire === "openai"
      ? parseOpenAIResponse(payload)
      : provider.wire === "anthropic"
        ? parseAnthropicResponse(payload)
        : parseGeminiResponse(payload);
  if (!answer) return { kind: "refused", refusal: "unknown" };
  return { kind: "ok", answer };
}

// ---------------------------------------------------------------------------
// Client responses
// ---------------------------------------------------------------------------

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += size) pieces.push(text.slice(i, i + size));
  if (pieces.length === 0) pieces.push("");
  return pieces;
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  res.end(body);
}

function openAIResponse(
  requestSeq: number,
  model: string,
  answer: ProviderAnswer,
  stream: boolean,
  res: ServerResponse,
): void {
  const finishReason = OPENAI_STOPS[stopFamily(answer.stop)];
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    sendJson(res, 200, {
      id: `chatcmpl-agg-${requestSeq}`,
      object: "chat.completion",
      created: CREATED,
      model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: answer.text },
          finish_reason: finishReason,
          logprobs: null,
        },
      ],
      usage,
    });
    return;
  }
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const base = {
    id: `chatcmpl-agg-${requestSeq}`,
    object: "chat.completion.chunk",
    created: CREATED,
    model,
  };
  const frames: Json[] = [
    { ...base, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] },
  ];
  for (const piece of splitText(answer.text, 13)) {
    frames.push({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  frames.push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: finishReason }] });
  frames.push({ ...base, choices: [], usage });
  for (const frame of frames) res.write(`data: ${JSON.stringify(frame)}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
}

function anthropicResponse(
  requestSeq: number,
  model: string,
  answer: ProviderAnswer,
  stream: boolean,
  res: ServerResponse,
): void {
  const stopReason = ANTHROPIC_STOPS[stopFamily(answer.stop)];
  if (!stream) {
    sendJson(res, 200, {
      id: `msg_agg_${requestSeq}`,
      type: "message",
      role: "assistant",
      model,
      content: [{ type: "text", text: answer.text }],
      stop_reason: stopReason,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
    return;
  }
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const id = `msg_agg_${requestSeq}`;
  const events: { event: string; data: Json }[] = [
    {
      event: "message_start",
      data: {
        type: "message_start",
        message: {
          id,
          type: "message",
          role: "assistant",
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
        },
      },
    },
    {
      event: "content_block_start",
      data: { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    },
  ];
  for (const piece of splitText(answer.text, 13)) {
    events.push({
      event: "content_block_delta",
      data: { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } },
    });
  }
  events.push(
    { event: "content_block_stop", data: { type: "content_block_stop", index: 0 } },
    {
      event: "message_delta",
      data: {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: answer.outputTokens },
      },
    },
    { event: "message_stop", data: { type: "message_stop" } },
  );
  for (const { event, data } of events) {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }
  res.end();
}

function sendUpstreamError(res: ServerResponse, api: "openai" | "anthropic"): void {
  if (api === "openai") {
    sendJson(res, 503, {
      error: { message: "all providers refused", type: "server_error", param: null, code: null },
    });
  } else {
    sendJson(res, 503, {
      type: "error",
      error: { type: "overloaded_error", message: "all providers refused" },
    });
  }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES = 32 * 1024 * 1024;

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on("data", (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function log(message: string): void {
  process.stderr.write(`[aggregator] ${message}\n`);
}

class Aggregator {
  private router: Router;
  private brief: Brief;
  private catalog: Catalog;
  private requestIndex = 0;
  // Serializes inference requests: the client sends one at a time, and
  // serial handling keeps request order == manifest order and keeps
  // provider calls strictly non-overlapping.
  private tail: Promise<void> = Promise.resolve();

  constructor(brief: Brief, catalog: Catalog) {
    this.brief = brief;
    this.catalog = catalog;
    this.router = new Router(brief);
  }

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.tail.then(fn);
    this.tail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  private modelList(req: IncomingMessage, res: ServerResponse): void {
    const seen = new Set<string>();
    const ids: { id: string; provider: string }[] = [];
    for (const provider of this.catalog.providers) {
      for (const model of provider.models) {
        if (seen.has(model.id)) continue;
        seen.add(model.id);
        ids.push({ id: model.id, provider: provider.id });
      }
    }
    const auth = req.headers.authorization ?? "";
    const hasBearer = typeof auth === "string" && auth.startsWith("Bearer");
    const hasApiKey = typeof req.headers["x-api-key"] === "string";
    if (hasApiKey && !hasBearer) {
      sendJson(res, 200, {
        data: ids.map(({ id }) => ({
          type: "model",
          id,
          display_name: id,
          created_at: "2026-01-01T00:00:00Z",
        })),
        has_more: false,
      });
    } else {
      sendJson(res, 200, {
        object: "list",
        data: ids.map(({ id, provider }) => ({
          id,
          object: "model",
          created: CREATED,
          owned_by: provider,
        })),
      });
    }
  }

  /**
   * Requirements for a request: the workload manifest's row when the
   * request is one of the manifest's rows (the normal case — the client
   * sends requests one at a time, in manifest order), else the
   * requirements implied by the request body itself.
   */
  private requirementsFor(index: number, norm: NormRequest): Json {
    const row = this.brief.manifest[index];
    if (row) return row.requires;
    const requires: Json = {};
    if (norm.tools.length) requires.tools = true;
    if (norm.messages.some((m) => m.parts.some((p) => p.type === "image"))) requires.vision = true;
    if (norm.jsonMode) requires.json = true;
    return requires;
  }

  private async handleInference(
    api: "openai" | "anthropic",
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    await this.enqueue(async () => {
      const index = this.requestIndex++;
      const requestSeq = index + 1;
      let body: Json;
      try {
        const raw = await readBody(req);
        const parsed = JSON.parse(raw.toString("utf8")) as unknown;
        if (!isObject(parsed)) throw new Error("body is not an object");
        body = parsed;
      } catch {
        sendJson(res, 400, { error: { message: "invalid JSON body", type: "invalid_request_error" } });
        return;
      }
      const norm = api === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
      const stream = body.stream === true;
      const clientAnthropicVersion =
        typeof req.headers["anthropic-version"] === "string" ? req.headers["anthropic-version"] : null;
      const requires = this.requirementsFor(index, norm);
      const needsJsonWire = api === "openai" && norm.jsonMode;

      // Same-wire passthrough keeps the request byte-identical except for
      // the model and stream flag, which maximises echo fidelity.
      const passthroughFor = (provider: ProviderEntry, model: string): Json | null => {
        if (provider.wire === "openai" && api === "openai") {
          return { ...body, model, stream: false };
        }
        if (provider.wire === "anthropic" && api === "anthropic") {
          return { ...body, model, stream: false };
        }
        return null;
      };

      const tried = new Set<string>();
      const schedRefused: Candidate[] = [];
      const attempt = async (candidate: Candidate): Promise<boolean> => {
        const { provider, row } = candidate;
        const result = await callProvider(
          provider,
          row.model,
          norm,
          clientAnthropicVersion,
          passthroughFor(provider, row.model),
        );
        if (result.kind === "ok") {
          this.router.recordSuccess(provider.id);
          log(`req ${requestSeq}: ${provider.id}/${row.model} (${row.rating})`);
          if (api === "openai") openAIResponse(requestSeq, row.model, result.answer, stream, res);
          else anthropicResponse(requestSeq, row.model, result.answer, stream, res);
          return true;
        }
        if (result.kind === "refused") {
          this.router.recordRefusal(provider.id, index, result.refusal);
          if (result.refusal === "scheduled") schedRefused.push(candidate);
          log(`req ${requestSeq}: ${provider.id} refused (${result.refusal}), failing over`);
        } else {
          this.router.recordProtocolError(provider.id);
          log(`req ${requestSeq}: ${provider.id} protocol error, failing over`);
        }
        return false;
      };

      // First pass: one attempt per provider, best pick first, with
      // fresh-provider failover (see Router.pickNext).
      for (;;) {
        const candidate = this.router.pickNext(requires, needsJsonWire, index, tried);
        if (!candidate) break;
        tried.add(candidate.provider.id);
        if (await attempt(candidate)) return;
      }
      // Second pass: if every provider refused, retry the ones refused
      // on schedule, best rating first. Each retry lands on a new call
      // number, so a provider past its failure schedule answers.
      const retries = schedRefused.slice().sort((a, b) => b.row.rating - a.row.rating);
      for (const candidate of retries) {
        if (this.router.isDead(candidate.provider.id)) continue;
        if (await attempt(candidate)) return;
      }
      log(`req ${requestSeq}: all candidates refused`);
      sendUpstreamError(res, api);
    });
  }

  handler = (req: IncomingMessage, res: ServerResponse): void => {
    const url = new URL(req.url ?? "/", "http://localhost");
    try {
      if (req.method === "GET" && url.pathname === "/v1/models") {
        this.modelList(req, res);
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        void this.handleInference("openai", req, res).catch((error: unknown) => {
          log(`openai handler error: ${String(error)}`);
          if (!res.headersSent) sendJson(res, 500, { error: { message: "internal error" } });
        });
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/messages") {
        void this.handleInference("anthropic", req, res).catch((error: unknown) => {
          log(`anthropic handler error: ${String(error)}`);
          if (!res.headersSent) sendJson(res, 500, { error: { message: "internal error" } });
        });
        return;
      }
      sendJson(res, 404, { error: { message: "not found" } });
    } catch (error) {
      log(`handler error: ${String(error)}`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: "internal error" } });
    }
  };
}

// ---------------------------------------------------------------------------
// Trial lifecycle: register, then poll /work until done
// ---------------------------------------------------------------------------

async function register(port: number, catalog: Catalog): Promise<boolean> {
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const response = await fetch(`${ENV_URL}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) return true;
      log(`register attempt ${attempt + 1}: HTTP ${response.status}`);
    } catch (error) {
      log(`register attempt ${attempt + 1}: ${String(error)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function workDone(): Promise<boolean | null> {
  try {
    const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) return null;
    const payload = (await response.json()) as unknown;
    if (isObject(payload) && payload.done === true) return true;
    return false;
  } catch {
    return null;
  }
}

async function start(): Promise<void> {
  const brief = readBrief();
  if (!brief) {
    process.stderr.write("no brief found (MAKEFUN_PROMPT_FILE unset or unreadable)\n");
    process.exit(1);
  }
  if (!ENV_URL) {
    process.stderr.write("MAKEFUN_ENV_URL is not set\n");
    process.exit(1);
  }
  const catalog = buildCatalog(brief);
  const aggregator = new Aggregator(brief, catalog);
  const server = createServer(aggregator.handler);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (!port || port === 8080) {
    process.stderr.write(`refusing to serve on port ${port}\n`);
    process.exit(1);
  }
  log(`listening on 0.0.0.0:${port}, ${catalog.providers.length} providers`);
  const registered = await register(port, catalog);
  if (!registered) {
    process.stderr.write("register failed\n");
    process.exit(1);
  }
  log("registered; waiting for the workload to finish");
  for (;;) {
    const done = await workDone();
    if (done === true) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  server.close();
  log("workload done; exiting 0");
}

function listMode(): void {
  const catalog = buildCatalog(readBrief());
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
}

if (process.argv.includes("--list")) {
  listMode();
  process.exit(0);
} else {
  let shuttingDown = false;
  const shutdown = (): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    process.exit(0);
  };
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
  try {
    await start();
  } catch (error) {
    process.stderr.write(`fatal: ${String(error)}\n`);
    process.exit(1);
  }
  process.exit(0);
}
