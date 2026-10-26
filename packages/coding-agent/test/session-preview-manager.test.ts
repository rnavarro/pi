import { describe, expect, it } from "vitest";
import { SessionManager } from "../src/core/session-manager.ts";

function userMessage(text: string) {
	return { role: "user" as const, content: text, timestamp: 1 };
}

function assistantMessage(text: string) {
	return {
		role: "assistant" as const,
		content: [{ type: "text" as const, text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop" as const,
		timestamp: 1,
	};
}

function sessionWithTurns(turns: number): SessionManager {
	const manager = SessionManager.inMemory();
	for (let i = 0; i < turns; i++) {
		manager.appendMessage(userMessage(`question ${i}`));
		manager.appendMessage(assistantMessage(`answer ${i}`));
	}
	return manager;
}

/** The construction the boundary preview used to perform. */
function legacyPreviewOf(parent: SessionManager): SessionManager {
	const header = parent.getHeader();
	if (!header) throw new Error("header missing");
	return SessionManager.inMemory(process.cwd(), undefined, [header, ...parent.getBranch()]);
}

describe("SessionManager.preview", () => {
	it("produces the same projection as re-loading the branch", () => {
		const parent = sessionWithTurns(4);
		const now = SessionManager.preview(parent).buildSessionProjection();
		const before = legacyPreviewOf(parent).buildSessionProjection();

		expect(JSON.stringify(now.messages)).toBe(JSON.stringify(before.messages));
		expect(now.thinkingLevel).toBe(before.thinkingLevel);
		expect(now.model).toEqual(before.model);
	});

	it("matches the old construction even when the session has an abandoned branch", () => {
		// The preview used to hold only the current branch; sharing the parent's
		// index brings off-path entries along. The projection follows parents from
		// the leaf, so it must not change.
		const parent = sessionWithTurns(3);
		const entries = parent.getEntries();
		const firstUser = entries.find((entry) => entry.type === "message" && entry.message.role === "user");
		expect(firstUser).toBeDefined();
		parent.branch(firstUser!.id);
		parent.appendMessage(userMessage("a different second question"));
		parent.appendMessage(assistantMessage("a different answer"));

		const now = SessionManager.preview(parent).buildSessionProjection();
		const before = legacyPreviewOf(parent).buildSessionProjection();
		expect(JSON.stringify(now.messages)).toBe(JSON.stringify(before.messages));
	});

	it("does not write through to the parent", () => {
		const parent = sessionWithTurns(2);
		const parentCount = parent.getEntryCount();
		const parentProjection = parent.buildSessionProjection();

		const preview = SessionManager.preview(parent);
		preview.appendMessage(userMessage("only in the preview"));

		expect(parent.getEntryCount()).toBe(parentCount);
		// Compare content, not identity: projection identity is only stable once the
		// memoisation patch is in the tree, and this branch must stand on its own.
		expect(JSON.stringify(parent.buildSessionProjection().messages)).toBe(JSON.stringify(parentProjection.messages));
		expect(preview.getEntryCount()).toBe(parentCount + 1);
		expect(JSON.stringify(preview.buildSessionProjection().messages)).toContain("only in the preview");
	});

	it("applies a draft to the preview without disturbing the parent", () => {
		const parent = sessionWithTurns(2);
		const target = parent
			.getEntries()
			.find((entry) => entry.type === "message" && entry.message.role === "assistant");
		expect(target).toBeDefined();

		const preview = SessionManager.preview(parent);
		preview.appendContextEdit(target!.id, { content: "edited in the preview" } as never);

		expect(JSON.stringify(preview.buildSessionProjection().messages)).toContain("edited in the preview");
		expect(JSON.stringify(parent.buildSessionProjection().messages)).not.toContain("edited in the preview");
	});

	it("costs less than re-loading and indexing the session", () => {
		// The win is native map copies instead of an entry-by-entry index rebuild.
		// Compare the two constructions over the same session so machine speed
		// cancels out.
		const parent = sessionWithTurns(500);
		const time = (build: () => unknown, iterations: number) => {
			const started = performance.now();
			for (let i = 0; i < iterations; i++) build();
			return performance.now() - started;
		};

		// Warm both so neither pays first-call costs.
		SessionManager.preview(parent);
		legacyPreviewOf(parent);

		const legacy = time(() => legacyPreviewOf(parent), 40);
		const preview = time(() => SessionManager.preview(parent), 40);

		expect(preview).toBeLessThan(legacy);
	});
});
