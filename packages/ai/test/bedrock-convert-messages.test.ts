import { Type } from "typebox";
import { describe, expect, it, vi } from "vitest";

const bedrockMock = vi.hoisted(() => ({
	constructorCalls: [] as Array<Record<string, unknown>>,
	streamEvents: undefined as unknown[] | undefined,
}));

vi.mock("@aws-sdk/client-bedrock-runtime", () => {
	class BedrockRuntimeServiceException extends Error {}

	class BedrockRuntimeClient {
		constructor(config: Record<string, unknown>) {
			bedrockMock.constructorCalls.push(config);
		}

		send(): Promise<unknown> {
			if (bedrockMock.streamEvents) {
				const events = bedrockMock.streamEvents;
				return Promise.resolve({
					$metadata: { httpStatusCode: 200 },
					stream: (async function* () {
						yield* events;
					})(),
				});
			}
			return Promise.reject(new Error("mock send"));
		}
	}

	class ConverseStreamCommand {
		readonly input: unknown;

		constructor(input: unknown) {
			this.input = input;
		}
	}

	return {
		BedrockRuntimeClient,
		BedrockRuntimeServiceException,
		ConverseStreamCommand,
		StopReason: {
			END_TURN: "end_turn",
			STOP_SEQUENCE: "stop_sequence",
			MAX_TOKENS: "max_tokens",
			MODEL_CONTEXT_WINDOW_EXCEEDED: "model_context_window_exceeded",
			TOOL_USE: "tool_use",
		},
		CachePointType: { DEFAULT: "default" },
		CacheTTL: { ONE_HOUR: "ONE_HOUR" },
		ConversationRole: { ASSISTANT: "assistant", USER: "user" },
		ImageFormat: { JPEG: "jpeg", PNG: "png", GIF: "gif", WEBP: "webp" },
		ToolResultStatus: { ERROR: "error", SUCCESS: "success" },
	};
});

import { stream as streamBedrock } from "../src/api/bedrock-converse-stream.ts";
import type { Context, Message, Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const baseModel: Model<"bedrock-converse-stream"> = {
	id: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
	name: "Claude Sonnet 4.5 (US)",
	api: "bedrock-converse-stream",
	provider: "amazon-bedrock",
	baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
	contextWindow: 200000,
	maxTokens: 64000,
	compat: { supportsStrictMode: true },
};

const novaModel: Model<"bedrock-converse-stream"> = {
	...baseModel,
	id: "amazon.nova-lite-v1:0",
	name: "Nova Lite",
	reasoning: false,
	compat: undefined,
};

async function capturePayload(context: Context, model = baseModel): Promise<unknown> {
	let capturedPayload: unknown;
	const s = streamBedrock(model, normalizeContext(context), {
		cacheRetention: "none",
		signal: AbortSignal.abort(),
		onPayload: (payload) => {
			capturedPayload = payload;
			return payload;
		},
	});
	for await (const event of s) {
		if (event.type === "error") break;
	}
	return capturedPayload;
}

describe("Bedrock constrained sampling", () => {
	it("gates native strict tool use by model capability", async () => {
		const context: Context = {
			messages: [{ role: "user", content: "Use the tool", timestamp: Date.now() }],
			tools: [
				{
					name: "lookup",
					description: "Look up a value",
					parameters: Type.Object({ value: Type.String() }),
					constrainedSampling: { type: "json_schema", strict: "require" },
				},
			],
		};
		const payload = await capturePayload(context);
		const toolConfig = (payload as { toolConfig: { tools: Array<{ toolSpec: { strict?: boolean } }> } }).toolConfig;
		expect(toolConfig.tools[0].toolSpec.strict).toBe(true);

		context.tools![0].constrainedSampling = { type: "json_schema", strict: "prefer" };
		const novaPayload = await capturePayload(context, novaModel);
		const novaToolConfig = (
			novaPayload as {
				toolConfig: { tools: Array<{ toolSpec: { strict?: boolean } }> };
			}
		).toolConfig;
		expect(novaToolConfig.tools[0].toolSpec.strict).toBeUndefined();
	});
});

describe("Bedrock tool arguments", () => {
	it("preserves empty property names in streamed tool arguments", async () => {
		bedrockMock.streamEvents = [
			{ messageStart: { role: "assistant" } },
			{
				contentBlockStart: {
					contentBlockIndex: 0,
					start: { toolUse: { toolUseId: "tool-1", name: "edit" } },
				},
			},
			{
				contentBlockDelta: {
					contentBlockIndex: 0,
					delta: {
						toolUse: {
							input: '{"path":"/workspace/foobar/file.js","edits":[{"oldText":"first","newText":"updated first"},{"oldText":"second","newText":"updated second","":""}]}',
						},
					},
				},
			},
			{ contentBlockStop: { contentBlockIndex: 0 } },
			{ messageStop: { stopReason: "tool_use" } },
		];

		try {
			const message = await streamBedrock(
				baseModel,
				normalizeContext({ messages: [{ role: "user", content: "Use the tool", timestamp: Date.now() }] }),
				{ cacheRetention: "none" },
			).result();

			expect(message.content[0]).toEqual({
				type: "toolCall",
				id: "tool-1",
				name: "edit",
				arguments: {
					path: "/workspace/foobar/file.js",
					edits: [
						{ oldText: "first", newText: "updated first" },
						{ oldText: "second", newText: "updated second", "": "" },
					],
				},
			});
		} finally {
			bedrockMock.streamEvents = undefined;
		}
	});
});

// Regression for #8643: GPT models on Bedrock reject images inside toolResult.content.
describe("Bedrock tool-result images", () => {
	const firstImage = { type: "image" as const, mimeType: "image/png", data: "AQID" };
	const secondImage = { type: "image" as const, mimeType: "image/png", data: "BAUG" };
	const messages: Message[] = [
		{
			role: "toolResult",
			toolCallId: "read-1",
			toolName: "read",
			content: [{ type: "text", text: "first image" }, firstImage],
			isError: false,
			timestamp: 1,
		},
		{
			role: "toolResult",
			toolCallId: "read-2",
			toolName: "read",
			content: [secondImage],
			isError: true,
			timestamp: 2,
		},
	];

	it.each([
		{ id: "global.openai.gpt-6-sol", name: "GPT-6 Sol (Global)" },
		{ id: "us.openai.gpt-6-luna", name: "GPT-6 Luna (US)" },
		{ id: "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6-sol-v1:0", name: "GPT-6 Sol" },
		{ id: "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example", name: "GPT-6 Sol (App)" },
	])("hoists images after grouped tool results for $id", async ({ id, name }) => {
		const model = { ...baseModel, id, name };
		const payload = (await capturePayload({ messages }, model)) as {
			messages: Array<{ role: string; content: Array<Record<string, unknown>> }>;
		};
		const content = payload.messages[0].content;
		expect(payload.messages).toHaveLength(1);
		expect(content).toEqual([
			{ toolResult: { toolUseId: "read-1", content: [{ text: "first image" }], status: "success" } },
			{ toolResult: { toolUseId: "read-2", content: [{ text: "<empty>" }], status: "error" } },
			{ image: { format: "png", source: { bytes: Uint8Array.from([1, 2, 3]) } } },
			{ image: { format: "png", source: { bytes: Uint8Array.from([4, 5, 6]) } } },
		]);
		expect(messages[0].content).toEqual([{ type: "text", text: "first image" }, firstImage]);
	});

	it("keeps Claude tool-result images nested", async () => {
		const payload = (await capturePayload({ messages }, baseModel)) as {
			messages: Array<{ content: Array<Record<string, unknown>> }>;
		};
		const content = payload.messages[0].content;
		expect(content).toHaveLength(2);
		expect(content[0]).toEqual({
			toolResult: {
				toolUseId: "read-1",
				content: [
					{ text: "first image" },
					{ image: { format: "png", source: { bytes: Uint8Array.from([1, 2, 3]) } } },
				],
				status: "success",
			},
		});
		expect(content[1]).toEqual({
			toolResult: {
				toolUseId: "read-2",
				content: [{ image: { format: "png", source: { bytes: Uint8Array.from([4, 5, 6]) } } }],
				status: "error",
			},
		});
	});

	it("keeps directly attached user images in user content", async () => {
		const payload = (await capturePayload(
			{ messages: [{ role: "user", content: [firstImage], timestamp: 1 }] },
			{ ...baseModel, id: "global.openai.gpt-6-sol" },
		)) as { messages: Array<{ content: unknown[] }> };
		expect(payload.messages[0].content).toEqual([
			{ image: { format: "png", source: { bytes: Uint8Array.from([1, 2, 3]) } } },
		]);
	});
});

describe("bedrock convertMessages skips unknown content types", () => {
	it("skips unknown user content blocks instead of throwing", async () => {
		const messages: Message[] = [
			{
				role: "user",
				content: [
					{ type: "text", text: "hello" },
					{ type: "unknown", data: "foo" },
				] as any,
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toHaveLength(1);
		expect(p.messages[0].content[0]).toEqual({ text: "hello" });
	});

	it("skips unknown assistant content blocks instead of throwing", async () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "hello" },
					{ type: "unknown", data: "foo" },
				] as any,
				api: "bedrock-converse-stream",
				provider: "amazon-bedrock",
				model: baseModel.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toHaveLength(1);
		expect(p.messages[0].content[0]).toEqual({ text: "hello" });
	});

	it("replaces user messages with only unknown content blocks with a placeholder", async () => {
		const messages: Message[] = [
			{
				role: "user",
				content: [{ type: "unknown", data: "foo" }] as any,
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toEqual([{ text: "<empty>" }]);
	});

	it("replaces blank user string content with a placeholder", async () => {
		const payload = await capturePayload({
			messages: [{ role: "user", content: "   ", timestamp: Date.now() }],
		});
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toEqual([{ text: "<empty>" }]);
	});

	it("filters blank user text blocks when other content remains", async () => {
		const payload = await capturePayload({
			messages: [
				{
					role: "user",
					content: [
						{ type: "text", text: "" },
						{ type: "text", text: "hello" },
					],
					timestamp: Date.now(),
				},
			],
		});
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toEqual([{ text: "hello" }]);
	});

	it("replaces user content emptied by surrogate sanitization with a placeholder", async () => {
		const payload = await capturePayload({
			messages: [{ role: "user", content: String.fromCharCode(0xd83d), timestamp: Date.now() }],
		});
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content).toEqual([{ text: "<empty>" }]);
	});

	it("skips assistant text blocks emptied by surrogate sanitization", async () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: [{ type: "text", text: String.fromCharCode(0xd83d) }],
				api: "bedrock-converse-stream",
				provider: "amazon-bedrock",
				model: baseModel.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(0);
	});

	it("replaces blank tool result content with a placeholder", async () => {
		const messages: Message[] = [
			{
				role: "toolResult",
				toolCallId: "tool-1",
				toolName: "tool",
				content: [{ type: "text", text: "" }],
				isError: false,
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as {
			messages: Array<{ role: string; content: Array<{ toolResult: { content: unknown[] } }> }>;
		};
		expect(p.messages).toHaveLength(1);
		expect(p.messages[0].content[0].toolResult.content).toEqual([{ text: "<empty>" }]);
	});

	it("skips assistant messages with only unknown content blocks", async () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: [{ type: "unknown", data: "foo" }] as any,
				api: "bedrock-converse-stream",
				provider: "amazon-bedrock",
				model: baseModel.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			},
		];
		const payload = await capturePayload({ messages });
		expect(payload).toBeDefined();
		const p = payload as { messages: Array<{ role: string; content: unknown[] }> };
		expect(p.messages).toHaveLength(0);
	});

	it("removes empty property names only from replayed Bedrock input", async () => {
		const toolArguments: { path: string; edits: Array<Record<string, string>> } = {
			path: "/workspace/foobar/file.js",
			edits: [
				{ oldText: "first", newText: "updated first" },
				{ oldText: "second", newText: "updated second", "": "" },
			],
		};
		const messages: Message[] = [
			{
				role: "assistant",
				content: [
					{
						type: "toolCall",
						id: "tool-1",
						name: "edit",
						arguments: toolArguments,
					},
				],
				api: "bedrock-converse-stream",
				provider: "amazon-bedrock",
				model: baseModel.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "toolUse",
				timestamp: Date.now(),
			},
			{
				role: "toolResult",
				toolCallId: "tool-1",
				toolName: "edit",
				content: [{ type: "text", text: "done" }],
				isError: false,
				timestamp: Date.now(),
			},
			{ role: "user", content: "Continue", timestamp: Date.now() },
		];

		const payload = await capturePayload({ messages });
		const p = payload as {
			messages: Array<{ content: Array<{ toolUse?: { input: unknown } }> }>;
		};
		expect(p.messages[0].content[0].toolUse?.input).toEqual({
			path: "/workspace/foobar/file.js",
			edits: [
				{ oldText: "first", newText: "updated first" },
				{ oldText: "second", newText: "updated second" },
			],
		});
		expect(toolArguments.edits[1]).toEqual({ oldText: "second", newText: "updated second", "": "" });
	});
});
